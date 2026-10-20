"""Progress redraws must not hide the exception in durable job evidence."""

from spark_sweep.smoke_job import _tail


def test_error_before_thousands_of_tqdm_redraws_is_retained() -> None:
    error = "TRELLIS.2 produced an invalid GLB artifact: bad embedded image"
    progress = "\r\x1b[AGathering results from xatlas: 100%|████| 75649/75649 [00:00<00:00, 126434it/s]"
    assert _tail(error + "\n" + progress * 1000) == error


def test_error_after_concatenated_progress_keeps_only_diagnostics() -> None:
    progress = "Gathering results from xatlas: 100%|████| 75649/75649 [00:00<00:00, 126434it/s]"
    assert _tail(progress * 5 + "\nValueError: bad image") == "ValueError: bad image"


def test_non_progress_percentages_and_utf8_limit_are_preserved() -> None:
    assert _tail("ValueError: 100% of model inputs are invalid") == (
        "ValueError: 100% of model inputs are invalid"
    )
    assert len(_tail("é" * 2000, limit=31).encode("utf-8")) <= 31
    assert _tail("\rdecode: 100%|████| 2/2 [00:00<00:00, 10it/s]")
