from argparse import ArgumentParser
from pathlib import Path
import csv
from typing import List, Tuple, Union


def has_short_sessions(match: List[List[str]], min_session_length: int):
    hero_id = match[0][4]
    session = []

    for idx in range(len(match)):
        entry = match[idx]
        new_hero_id = entry[4]

        if hero_id != new_hero_id:
            if len(session) < min_session_length:
                return True
            hero_id = new_hero_id
            session = []
        else:
            session.append(entry[7])

    if len(session) < min_session_length:
        return True

    return False


def has_leaver(match: List[List[str]]):
    """
    Determines if one of the players has left the game early, or if an error resolving the information has occurred (leaver_status==-1) during the dataset construction.

    :param match: a list of rows that comprise a single match
    :return: true if at least one player left early, false otherwise.
    """
    for entry in match:
        if int(entry[3]) != 0:
            return True

    return False


def longer_than_min_duration(match: List[List[str]], min_duration: int):
    """
    Determines whether duration >= min_duration.
    :param match: a list of rows that comprise a single match
    :param min_duration: the minimum duration in seconds
    :return: true if the match is longer than the minimum duration, false otherwise.
    """
    return int(match[0][1]) >= min_duration


def get_next_match(reader) -> Tuple[List[List[str]], Union[List[str], None]]:
    match = []

    first_line = next(reader)
    match.append(first_line)

    match_id = first_line[0]

    for line in reader:
        next_match_id = line[0]

        if next_match_id != match_id:
            return match, line

        match.append(line)

    return match, None


def write_match(writer, match: List[List[str]]):
    writer.writerows(match)


def run(input_file_path: Path, output_file_path: Path, min_session_length: int = 2, min_duration: int = 900):
    with input_file_path.open("r") as input_file, output_file_path.open("w") as output_file:
        input_reader = csv.reader(input_file, delimiter="\t")
        output_writer = csv.writer(output_file, delimiter="\t")

        # copy header
        header = next(input_reader)
        output_writer.writerow(header)

        match, first_line_next_match = get_next_match(input_reader)
        if not has_short_sessions(match, min_session_length) and not has_leaver(match) and longer_than_min_duration(match, min_duration):
            write_match(output_writer, match)

        while first_line_next_match is not None:
            cache_first_line = first_line_next_match
            match, first_line_next_match = get_next_match(input_reader)

            match = [cache_first_line] + match
            if not has_short_sessions(match, min_session_length) and not has_leaver(match) and longer_than_min_duration(match, min_duration):
                write_match(output_writer, match)


if __name__ == "__main__":
    parser = ArgumentParser("dota-preprocessor", description="Will remove matches that contain sessions with less then the specified minimum session length.")
    parser.add_argument("input_file", type=str, help="input file in csv format")
    parser.add_argument("output_file", type=str, help="output file")
    parser.add_argument("--min_session_length", type=int, default=2, help="minimum session length (Default: 2)")
    parser.add_argument("--min_duration", type=int, default=900, help="minimum duration (Default: 900)")

    args = parser.parse_args()

    run(Path(args.input_file), Path(args.output_file), args.min_session_length, args.min_duration)
