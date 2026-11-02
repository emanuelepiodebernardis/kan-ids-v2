from pathlib import PurePosixPath, PureWindowsPath

from export_release import release_destination


def test_release_destination_is_separator_independent():
    expected = PurePosixPath('source/diagnostic_b/runner.py')
    posix = release_destination(
        PurePosixPath('adattamento-drift/review_extension/review3/runner.py')
    )
    windows = release_destination(
        PureWindowsPath(r'adattamento-drift\review_extension\review3\runner.py')
    )
    assert posix.as_posix() == expected.as_posix()
    assert windows.as_posix() == expected.as_posix()


def test_release_destination_routes_temporal_sources():
    destination = release_destination(
        PureWindowsPath(r'adattamento-drift\ton_temporal\replay.py')
    )
    assert destination.as_posix() == 'source/temporal/replay.py'


def test_release_destination_excludes_internal_markdown():
    assert release_destination(
        PureWindowsPath(r'adattamento-drift\review_extension\REVIEW_RESPONSE.md')
    ) is None
