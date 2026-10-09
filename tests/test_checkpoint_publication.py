import threading

import pytest

from archlab.automodel.checkpoint_publication import CheckpointPublisher


def test_background_publisher_overlaps_training_and_waits_before_exit():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    publisher = CheckpointPublisher()

    def publish():
        entered.set()
        assert release.wait(5)
        finished.set()

    publisher.submit(publish)
    assert entered.wait(5)
    assert not finished.is_set()
    publisher.check()
    release.set()
    publisher.close()
    assert finished.is_set()


def test_background_upload_error_is_never_silently_reported_complete():
    publisher = CheckpointPublisher()

    def fail():
        raise ValueError("checksum failed")

    publisher.submit(fail)
    with pytest.raises(ValueError, match="checksum"):
        publisher.close()
