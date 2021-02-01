def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'archive_path', type=str, default=None, nargs='?')
    parser.add_argument(
        '--dry', action='store_true', default=False)
    args = parser.parse_args()

    return args
