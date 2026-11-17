import pytest

from clippr.pipeline.download import DownloadError, is_local_source, validate_youtube_url


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
    "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://www.youtube.com/shorts/abc123",
    "https://www.youtube.com/live/abc123",
    "  https://www.youtube.com/watch?v=x  ",
])
def test_accepts_single_video_links(url):
    assert validate_youtube_url(url) == url.strip()


@pytest.mark.parametrize("url", [
    "ftp://www.youtube.com/watch?v=x",
    "https://evil.example.com/watch?v=x",
    "https://youtube.com.evil.example/watch?v=x",
    "https://www.youtube.com/watch",
    "https://www.youtube.com/playlist?list=PL123",
    "https://www.youtube.com/@somechannel",
    "https://youtu.be/",
    "file:///etc/passwd",
    "not a url",
])
def test_rejects_everything_else(url):
    with pytest.raises(DownloadError):
        validate_youtube_url(url)


def test_local_source_detection(fixture_video):
    assert is_local_source(str(fixture_video))
    assert not is_local_source("https://youtu.be/abc")
    assert not is_local_source("/no/such/file.mp4")
