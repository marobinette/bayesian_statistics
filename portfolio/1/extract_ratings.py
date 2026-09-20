#!/usr/bin/env python3
"""Stream a Lichess PGN dump -> CSV of one row per game with both players' ratings.

    python3 extract_ratings.py dump.pgn bong.csv
    python3 extract_ratings.py dump.pgn rapid.csv --speed rapid --opening "Sicilian Defense"
    zstdcat dump.pgn.zst | python3 extract_ratings.py - out.csv

Defaults to --speed blitz --opening "Bongcloud Attack". Games are selected on
the [Event] and [Opening] tags alone: Lichess has already classified the
deepest book position reached, so there is no need to parse movetext. Pass
--speed any or --opening any to drop either filter.
"""
import sys

# Lichess keeps a separate rating pool per speed, so mixing them in one column
# makes ratings incomparable across rows. Filter to one speed to avoid that.
SPEEDS = ("ultrabullet", "bullet", "blitz", "rapid", "classical", "correspondence")

DEFAULT_SPEED = "blitz"
DEFAULT_OPENING = "Bongcloud Attack"  # Lichess's [Opening] name for 1.e4 e5 2.Ke2

ANY = "any"


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


def opening_matches(opening, want, prefix):
    """True if an [Opening] value is `want` or a variation of it.

    Lichess names variations as 'Family: Variation, Sub-variation', so
    --opening "Sicilian Defense" takes the whole family while
    --opening "Bongcloud Attack" (which has no variations) takes just itself.
    """
    return opening == want or opening.startswith(prefix)


def extract(fin, fout, speed=None, opening_want=None):
    white = black = welo = belo = date = opening = b"?"
    scanned = written = 0
    started = keep = False
    want_speed = speed.encode() if speed else None
    want_open = opening_want.encode() if opening_want else None
    prefix = want_open + b":" if want_open else b""
    write = fout.write
    for line in fin:
        # Movetext and blank lines are most of the bytes; reject them on byte 0.
        if line[:1] != b"[":
            continue
        if line.startswith(b"[Event "):
            # Start of a record: flush the previous one, then reset, so a game
            # missing any tag can't inherit a value from the game before it.
            if started:
                if keep:
                    if want_open is None or opening_matches(opening, want_open, prefix):
                        # opening is quoted: 12% of names contain a comma.
                        write(b'%s,%s,%s,%s,%s,"%s"\n' % (
                            date, white, welo, black, belo, opening))
                        written += 1
                    scanned += 1
                    if scanned % 5000000 == 0:
                        print(f"  scanned {scanned//1000000}M, kept {written}",
                              file=sys.stderr, flush=True)
            white = black = welo = belo = date = opening = b"?"
            started = True
            keep = want_speed is None or event_speed(line) == want_speed
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
        if want_open is None or opening_matches(opening, want_open, prefix):
            write(b'%s,%s,%s,%s,%s,"%s"\n' % (
                date, white, welo, black, belo, opening))
            written += 1
        scanned += 1
    return scanned, written


def take_option(args, name, default):
    """Pop '--name value' out of args, returning value or the default."""
    if name not in args:
        return default
    i = args.index(name)
    if i + 1 >= len(args):
        sys.exit(f"{name} needs a value")
    value = args.pop(i + 1)
    args.pop(i)
    return value


def main():
    args = list(sys.argv[1:])
    speed = take_option(args, "--speed", DEFAULT_SPEED).lower()
    opening = take_option(args, "--opening", DEFAULT_OPENING)
    if speed != ANY and speed not in SPEEDS:
        # Catch the typo now, not after a 20-minute run produces 0 rows.
        sys.exit(f"unknown speed {speed!r}; expected one of {', '.join(SPEEDS)}, {ANY}")
    if len(args) != 2:
        sys.exit(__doc__)
    path, out = args
    speed = None if speed == ANY else speed
    opening = None if opening.lower() == ANY else opening
    fin = sys.stdin.buffer if path == "-" else open(path, "rb")
    with open(out, "wb") as fout:
        fout.write(b"date,white,white_elo,black,black_elo,opening\n")
        scanned, written = extract(fin, fout, speed, opening)
    print(f"scanned {scanned} games ({speed or 'all speeds'}), "
          f"wrote {written} ({opening or 'all openings'}) -> {out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
