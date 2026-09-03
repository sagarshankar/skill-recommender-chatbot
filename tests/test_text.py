from skill_recommender.text import clean_span, split_sentences


class TestSplitSentences:
    def test_splits_on_sentence_punctuation(self):
        assert split_sentences("Knows Python. Knows SQL!") == ["Knows Python.", "Knows SQL!"]

    def test_splits_on_newlines(self):
        assert split_sentences("Line one\nLine two") == ["Line one", "Line two"]

    def test_splits_on_leading_bullet_marker(self):
        assert split_sentences("• First item") == ["First item"]

    def test_only_first_of_consecutive_bulleted_lines_is_stripped(self):
        # Known quirk: once "\n+" matches at a line boundary, the regex
        # alternation takes it before trying the fuller bullet-marker
        # alternative, so later items in a bulleted list keep their "•".
        assert split_sentences("• First item\n• Second item") == ["First item", "• Second item"]

    def test_inserts_space_at_camel_case_boundary(self):
        # Scraped bullet lists sometimes lose all whitespace between items,
        # leaving only a lowercase->uppercase transition as a boundary signal.
        result = split_sentences("as assignedServices or replaces defective parts")
        assert "assigned Services" in " ".join(result)

    def test_empty_and_whitespace_only_input(self):
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []

    def test_strips_surrounding_whitespace(self):
        assert split_sentences("  Hello world.  ") == ["Hello world."]


class TestCleanSpan:
    def test_lowercases_and_strips_punctuation(self):
        assert clean_span("  Python.  ") == "python"

    def test_drops_wordpiece_fragments(self):
        assert clean_span("##igh school") is None

    def test_drops_spans_over_max_words(self):
        long_span = "this is a very long clause that goes way beyond the word limit"
        assert clean_span(long_span) is None

    def test_keeps_spans_at_max_words(self):
        five_words = "one two three four five"
        assert clean_span(five_words) == "one two three four five"

    def test_drops_empty_after_stripping(self):
        assert clean_span("   ...   ") is None

    def test_keeps_normal_short_span(self):
        assert clean_span("HVAC") == "hvac"
