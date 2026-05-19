from subtitles.schemas import SubtitleResult


def milliseconds_to_srt_time(milliseconds: int) -> str:
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def subtitle_result_to_srt(result:
                            SubtitleResult) -> str:
    blocks: list[str] = []

    for segment in result.segments:
        start_time = milliseconds_to_srt_time(segment.start_ms)
        end_time = milliseconds_to_srt_time(segment.end_ms)

        blocks.append(
            f"{segment.index}\n"
            f"{start_time} --> {end_time}\n"
            f"{segment.text}"
        )

    return "\n\n".join(blocks)