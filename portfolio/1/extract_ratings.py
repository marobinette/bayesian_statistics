#!/usr/bin/env python3
# Constructed using Claude and verified against lichess API - details in eda.ipynb

"""Stream a Lichess PGN dump -> CSV of one row per game with both players' ratings.

    python3 extract_ratings.py dump.pgn ratings.csv
    python3 extract_ratings.py dump.pgn rapid.csv --speed rapid
    python3 extract_ratings.py dump.pgn bong.csv --bongcloud-only   # 1.e4 e5 2.Ke2 only
    zstdcat dump.pgn.zst | python3 extract_ratings.py - out.csv --speed blitz

Only tag pairs are read, plus (with --bongcloud) the first ~200 bytes of movetext.
Games where White plays 2.Ke2 also get their full move sequence in the "moves"
column (SAN, space-separated, no move numbers), so you can see when -- or
whether -- Black answers with ...Ke7.

--bongcloud also emits Lichess's own [Opening] name, as an independent check
on the move scan: a game tagged "Bongcloud Attack" that the scan missed (or
vice versa) is reported as a mismatch at the end of the run.
"""
import sys

# Lichess keeps a separate rating pool per speed, so mixing them in one column
# makes ratings incomparable across rows. Filter to one speed to avoid that.
SPEEDS = ("ultrabullet", "bullet", "blitz", "rapid", "classical", "correspondence")

# Lichess's [Opening] name for 1.e4 e5 2.Ke2. It classifies the deepest book
# position reached, so it corresponds to "narrow" below -- never to "broad".
BA = b"Bongcloud Attack"


def event_speed(line):
    """Speed word from an Event tag, as a lowercase bytestring.

    Handles '[Event "Rated Blitz game"]' and the tournament/swiss forms, e.g.
    '[Event "Rated Blitz tournament https://lichess.org/tournament/x"]'.
    Tokenised, not substring-matched, so 'bullet' can't match 'UltraBullet'.
    """
    parts = line[8:-3].lower().split(b" ", 2)
    if len(parts) > 1 and parts[0] == b"rated":
        return parts[1]
    return parts[0] if parts else b""


def first_plies(head, n=4):
    """First n plies of a movetext head (all of them if n is None), comments
    and move numbers removed.

    Must survive three movetext forms seen in the dumps:
      1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Ke2   (clocks)
      1. e4 { [%eval 0.25] } 1... e5 { [%eval 0.2] } 2. Ke2        (analysed)
      1. e4 e5 2. Ke2                                              (no comments)
    The third has no '1...' markers at all, so substring matching on ' 2. Ke2'
    silently misses it. Annotations ('2. Ke2?', 'Ke2??') are stripped too.
    """
    parts, i = [], 0
    while True:
        j = head.find(b"{", i)
        if j < 0:
            parts.append(head[i:])
            break
        parts.append(head[i:j])
        k = head.find(b"}", j)
        if k < 0:
            break
        i = k + 1
    plies = []
    for t in b" ".join(parts).split():
        c = t[:1]
        if c.isdigit() or c == b"*":   # move numbers "1." "1..." and results
            continue
        plies.append(t.rstrip(b"?!+#"))
        if len(plies) == n:
            break
    return plies


def extract(fin, fout, speed=None, bongcloud=False, only=False):
    white = black = welo = belo = date = opening = b"?"
    e4e5 = wke2 = bke7 = b"0"
    moves = b""
    scanned = written = broad = narrow = doubles = tagged = mismatch = 0
    started = keep = False
    want = speed.encode() if speed else None
    write = fout.write
    for line in fin:
        # Movetext and blank lines are most of the bytes; reject them on byte 0.
        if line[:1] != b"[":
            if bongcloud and line[:2] == b"1.":
                # Ply 4 always lands within 200 bytes, even with both [%eval]
                # and [%clk] comments present.
                p = first_plies(line[:200])
                n = len(p)
                if n > 1 and p[0] == b"e4" and p[1] == b"e5":
                    e4e5 = b"1"
                if n > 2 and p[2] == b"Ke2":
                    wke2 = b"1"
                    # Only ~1 game in 3000 gets here, so parsing the whole
                    # line is cheap. Lichess puts all movetext on one line.
                    moves = b" ".join(first_plies(line, None))
                    if n > 3 and p[3] == b"Ke7":
                        bke7 = b"1"
            continue
        if line.startswith(b"[Event "):
            # Start of a record: flush the previous one, then reset, so a game
            # missing any tag can't inherit a value from the game before it.
            if started:
                if keep:
                    if wke2 == b"1":
                        broad += 1
                        if e4e5 == b"1":
                            narrow += 1
                            doubles += bke7 == b"1"
                    tagged += opening == BA
                    mismatch += (opening == BA) != (e4e5 == b"1" and wke2 == b"1")
                    if not (only and (wke2 == b"0" or e4e5 == b"0")):
                        if bongcloud:
                            # opening is quoted: 12% of names contain a comma.
                            write(b'%s,%s,%s,%s,%s,%s,%s,%s,"%s","%s"\n' % (
                                date, white, welo, black, belo, e4e5, wke2, bke7,
                                opening, moves))
                        else:
                            write(b"%s,%s,%s,%s,%s\n" % (
                                date, white, welo, black, belo))
                        written += 1
                    scanned += 1
                    if scanned % 5000000 == 0:
                        print(f"  scanned {scanned//1000000}M, kept {written}",
                              file=sys.stderr, flush=True)
            white = black = welo = belo = date = opening = b"?"
            e4e5 = wke2 = bke7 = b"0"
            moves = b""
            started = True
            keep = want is None or event_speed(line) == want
        elif line.startswith(b"[White "):
            white = line[8:-3]
        elif line.startswith(b"[Black "):
            black = line[8:-3]
        elif line.startswith(b"[WhiteElo "):
            welo = line[11:-3]
        elif line.startswith(b"[BlackElo "):
            belo = line[11:-3]
        elif line.startswith(b"[UTCDate "):
            date = line[10:-3]
        elif line.startswith(b"[Opening "):
            # '[Opening "' is ten bytes: tag name, space, opening quote.
            opening = line[10:-3]
    if started and keep:
        if wke2 == b"1":
            broad += 1
            if e4e5 == b"1":
                narrow += 1
                doubles += bke7 == b"1"
        tagged += opening == BA
        mismatch += (opening == BA) != (e4e5 == b"1" and wke2 == b"1")
        if not (only and (wke2 == b"0" or e4e5 == b"0")):
            if bongcloud:
                write(b'%s,%s,%s,%s,%s,%s,%s,%s,"%s","%s"\n' % (
                    date, white, welo, black, belo, e4e5, wke2, bke7, opening,
                    moves))
            else:
                write(b"%s,%s,%s,%s,%s\n" % (date, white, welo, black, belo))
            written += 1
        scanned += 1
    return scanned, written, broad, narrow, doubles, tagged, mismatch


def main():
    args = list(sys.argv[1:])
    only = "--bongcloud-only" in args
    if only:
        args.remove("--bongcloud-only")
    bongcloud = only or "--bongcloud" in args
    if "--bongcloud" in args:
        args.remove("--bongcloud")
    speed = None
    if "--speed" in args:
        i = args.index("--speed")
        speed = args.pop(i + 1).lower()
        args.pop(i)
        if speed not in SPEEDS:
            # Catch the typo now, not after a 20-minute run produces 0 rows.
            sys.exit(f"unknown speed {speed!r}; expected one of {', '.join(SPEEDS)}")
    if len(args) != 2:
        sys.exit(__doc__)
    path, out = args
    fin = sys.stdin.buffer if path == "-" else open(path, "rb")
    header = b"date,white,white_elo,black,black_elo"
    if bongcloud:
        header += b",e4e5,wke2,bke7,opening,moves"
    with open(out, "wb") as fout:
        fout.write(header + b"\n")
        scanned, written, broad, narrow, doubles, tagged, mismatch = extract(
            fin, fout, speed, bongcloud, only)
    print(f"scanned {scanned} games, wrote {written} ({speed or 'all speeds'}) -> {out}",
          file=sys.stderr)
    if bongcloud:
        # scanned is the denominator for a rate, so report it even when
        # --bongcloud-only throws every other row away. Only the narrow count
        # is comparable to the Explorer: 2.Ke2 is legal after 1.e4 anything,
        # so "broad" counts positions the Explorer query never sees.
        print(f"  2.Ke2 any reply : {broad}", file=sys.stderr)
        print(f"  1.e4 e5 2.Ke2   : {narrow}   <- matches Opening Explorer",
              file=sys.stderr)
        print(f"  ...2...Ke7      : {doubles}", file=sys.stderr)
        # Lichess's classifier and the move scan should agree exactly on the
        # narrow line. Any mismatch is a parser bug or an unexpected tag name,
        # and either way it means the narrow count is not trustworthy.
        print(f'  [Opening] "{BA.decode()}" : {tagged}', file=sys.stderr)
        if mismatch:
            print(f"  MISMATCH vs move scan : {mismatch}   <- investigate",
                  file=sys.stderr)


if __name__ == "__main__":
    main()