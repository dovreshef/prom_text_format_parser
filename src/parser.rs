use super::{
    Float,
    Label,
    Labels,
    Metric,
    Sample,
    Type,
    Value,
    ValueType,
};
use std::ops::Range;
use winnow::{
    ascii::{
        digit1,
        escaped,
        newline,
        space0,
        Caseless,
    },
    combinator::{
        alt,
        cut_err,
        delimited,
        eof,
        opt,
        preceded,
        repeat,
        separated,
        terminated,
    },
    error::ParseError,
    stream::{
        Accumulate,
        AsBStr,
        AsChar,
    },
    token::{
        none_of,
        one_of,
        tag,
        take_till,
        take_while,
    },
    PResult,
    Parser,
};

/// Parse a valid prometheus `metric_name` or `label_name`.
fn name_parser(input: &mut &str) -> PResult<String> {
    let start_group = ('a'..='z', 'A'..='Z', '_', ':');
    let rest_group = ('a'..='z', 'A'..='Z', '0'..='9', '_', ':');
    (one_of(start_group), take_while(0.., rest_group))
        .map(|(ch, rest)| format!("{ch}{rest}"))
        .parse_next(input)
}

/// Parse a valid prometheus label value.
///
/// Examples:
///
/// * `"Test"`
/// * `"Some value"`
fn parse_label_value(input: &mut &str) -> PResult<String> {
    let escaped = escaped(none_of(br#""\"#), '\\', one_of(br#""n\"#));
    preceded('\"', cut_err(terminated(escaped, '\"')))
        .parse_to()
        .parse_next(input)
}

/// Parse a Prometheus label key value pair.
///
/// Examples:
///
/// * `key1="value1"`
/// * `key = "value"`
/// * `key= "val"`
fn label_key_value_parser(input: &mut &str) -> PResult<(String, String)> {
    let key = name_parser.parse_next(input)?;
    let _ = (space0, '=', space0).parse_next(input)?;
    let val = parse_label_value.parse_next(input)?;
    Ok((key, val))
}

// Enable us to parse the collection of key value pairs into the `Labels` structure
impl Accumulate<(String, String)> for Labels {
    fn initial(capacity: Option<usize>) -> Self {
        Vec::with_capacity(capacity.unwrap_or(4)).into()
    }

    fn accumulate(&mut self, acc: (String, String)) {
        self.push(Label::new(acc.0, acc.1));
    }
}

/// Parses a list of labels delimited by braces
///
/// Examples:
/// * `{key1="value1",key2="value2"}`
/// * `{key1="value1", key2 = "value2"}`
/// * `{ key1="value1", key2 = "value2" }`
fn labels_parser(input: &mut &str) -> PResult<Labels> {
    let separator = (space0, ',', space0);
    let list = separated(1.., label_key_value_parser, separator);
    let start_delimiter = ("{", space0);
    let end_delimiter = (space0, "}");
    let mut labels = delimited(start_delimiter, list, end_delimiter);
    labels.parse_next(input)
}

/// Parse a valid Prometheus float value (+Inf, -Inf, NaN, ...)
fn float_value_parser(input: &mut &str) -> PResult<Float> {
    let number = (
        opt(one_of(['+', '-'])),
        alt((
            (digit1, opt(('.', opt(digit1)))).map(|_| ()),
            ('.', digit1).map(|_| ()),
        )),
        opt((one_of(['e', 'E']), opt(one_of(['+', '-'])), cut_err(digit1))),
    )
        .recognize();
    let nan = tag(Caseless("nan"));
    let inf = alt((tag(Caseless("inf")), tag(Caseless("+inf"))));
    let neg_inf = tag(Caseless("-inf"));
    alt((number, nan, inf, neg_inf))
        .parse_to()
        .parse_next(input)
}

/// Parse a valid Prometheus int value
fn int_value_parser(input: &mut &str) -> PResult<i64> {
    let prefix = opt(one_of(['+', '-']));
    (prefix, digit1).recognize().parse_to().parse_next(input)
}

/// Validate that the next character is either a new line or an EoF, returning an error if not
fn new_line_or_eof_parser(input: &mut &str) -> PResult<()> {
    alt((eof.map(|_| ()), newline.map(|_| ()))).parse_next(input)
}

/// Parse the rest of line until either EoF or NewLine (Parsing & ignoring the newline character)
fn rest_of_the_line_parser<'a>(input: &mut &'a str) -> PResult<&'a str> {
    let rest = preceded(space0, take_till(1.., AsChar::is_newline)).parse_next(input)?;
    new_line_or_eof_parser.parse_next(input)?;
    Ok(rest)
}

/// The four possible types of lines in the Prometheus exposition format
#[derive(Debug, Clone)]
enum Line {
    Empty,
    Comment(String),
    Help {
        name: String,
        desc: String,
    },
    Type {
        name: String,
        kind: Type,
    },
    Sample {
        name: String,
        labels: Labels,
        value: Value,
    },
}

/// Parse a Prometheus comment line.
///
/// A comment is anything that starts with #.
///
/// Example:
/// * `# This is a comment`
fn comment_line_parser(input: &mut &str) -> PResult<Line> {
    preceded((space0, tag("#"), space0), rest_of_the_line_parser)
        .map(|v| Line::Comment(v.into()))
        .parse_next(input)
}

/// Parse a Prometheus HELP line.
///
/// A HELP line is a comment that starts with #, followed by "HELP", followed by the name of
/// the metric, followed by its description.
///
/// Example:
/// * `# HELP http_request_duration_seconds A histogram of the request duration.`
fn help_line_parser(input: &mut &str) -> PResult<Line> {
    let ignored = (space0, tag("#"), space0, tag("HELP"), space0);
    let name = preceded(ignored, name_parser).parse_next(input)?;
    let desc = rest_of_the_line_parser.parse_to().parse_next(input)?;
    Ok(Line::Help { name, desc })
}

/// Parse a Prometheus TYPE line.
///
/// A TYPE line is a comment that starts with #, followed by "TYPE", followed by the name of
/// the metric, followed by its type - one of (counter, gauge, untyped, summary, histogram).
///
/// Example:
/// * `# TYPE http_request_duration_seconds histogram`
fn type_line_parser(input: &mut &str) -> PResult<Line> {
    let ignored = (space0, tag("#"), space0, tag("TYPE"), space0);
    let name = preceded(ignored, name_parser).parse_next(input)?;
    let kind = rest_of_the_line_parser.parse_to().parse_next(input)?;
    Ok(Line::Type { name, kind })
}

/// Parse a Prometheus metric line.
///
/// Returns:
/// The metric name and the `Value`
///
/// Examples:
/// * `data_sent:bytes{th_id="worker_0",type="duplex"} 1395`
/// * `metric_without_timestamp_and_labels 12.47`
/// * `metric_without_timestamp_and_labels 12.47 -1`
/// * `http_request_duration_seconds_count 144320`
fn sample_line_parser(input: &mut &str) -> PResult<Line> {
    let name = name_parser.parse_next(input)?;
    // Parse the labels, if they exist, otherwise return an empty Vec.
    let labels = preceded(space0, opt(labels_parser))
        .parse_next(input)?
        .unwrap_or_default();
    let value = preceded(space0, float_value_parser).parse_next(input)?;
    let timestamp = preceded(space0, opt(int_value_parser)).parse_next(input)?;
    // Expect the line to end after
    (space0, new_line_or_eof_parser).parse_next(input)?;
    // The value type is chosen as default here, and is modified based on the type and the name
    // later
    let value = Value::new(ValueType::Sample, value, timestamp);
    Ok(Line::Sample {
        name,
        labels,
        value,
    })
}

/// Parse an empty line. For completeness.
fn empty_line_parser(input: &mut &str) -> PResult<Line> {
    (space0, newline).map(|_| Line::Empty).parse_next(input)
}

/// Parse a Prometheus metric.
///
/// Composed of one of the four possible metric lines
fn metric_line_parser(input: &mut &str) -> PResult<Line> {
    alt((
        help_line_parser,
        type_line_parser,
        comment_line_parser,
        sample_line_parser,
        empty_line_parser,
    ))
    .parse_next(input)
}

/// Parse a complete scrape into its low level composing lines.
///
/// This function is used by the higher level parser to parse each line in the scrape into
/// a list of lines.
/// Each metric will later be composed of multiple lines.
fn scrape_lines_parser(input: &mut &str) -> PResult<Vec<Line>> {
    repeat(1.., metric_line_parser).parse_next(input)
}

/// The errors that can result from failure to parse.
/// Either failure to parse the lines, or failure to assemble the metrics.
#[derive(Debug, Clone, derive_more::From)]
pub enum ScrapeParseError {
    /// Error occurred at the line parsing stage
    Parse(String),
    /// Failed to coalesce some lines into metrics
    Collect(Vec<MetricError>),
}

impl<I, E> From<ParseError<I, E>> for ScrapeParseError
where
    I: AsBStr,
    E: std::fmt::Display,
{
    fn from(value: ParseError<I, E>) -> Self {
        Self::Parse(value.to_string())
    }
}

/// A failure to assemble multiple lines into a metric.
/// Composed of the line the error occurred and the error message.
#[derive(Debug, Clone, derive_more::Constructor)]
pub struct MetricError {
    /// The line number where the error occurred
    pub line_no: Range<usize>,
    /// The error string
    pub reason: String,
}

#[derive(Debug, Clone, derive_more::Display)]
enum MetricState {
    // Initial state
    #[display(fmt = "start")]
    Start,
    // From this point in the parsing, we have the metric name
    #[display(fmt = "help ({_0})")]
    Help(String),
    #[display(fmt = "type ({_0})")]
    Type(String),
    #[display(fmt = "sample ({_0})")]
    Sample(String),
}

/// Assemble Metric from the scrape lines
#[derive(Debug)]
struct MetricAssembler {
    /// Lines to process
    lines: Vec<Line>,
    /// Current line - for debugging
    current: usize,
}

impl MetricAssembler {
    fn new(mut lines: Vec<Line>) -> Self {
        // So we can always pop the last line
        lines.reverse();
        Self { lines, current: 0 }
    }

    /// Return the last line to the lines buffer
    fn rewind(&mut self, put_back: Line) {
        // put back the new line
        self.lines.push(put_back);
        // rewind the line pointer
        self.current -= 1;
    }
}

impl Iterator for MetricAssembler {
    /// Either the parsed metric or the metric error
    type Item = Result<Metric, MetricError>;

    /// Assemble a single metric from the remaining scrape lines
    fn next(&mut self) -> Option<Self::Item> {
        let mut maybe_kind = None;
        let mut maybe_desc = None;
        let mut samples = Vec::new();
        let mut state = MetricState::Start;

        loop {
            self.current += 1;
            state = match (state, self.lines.pop()) {
                // Skip empty/comment lines
                (state, Some(Line::Empty | Line::Comment(_))) => state,
                // Exit case
                (MetricState::Start, None) => {
                    return None;
                }
                (MetricState::Start, Some(Line::Help { name, desc })) => {
                    maybe_desc = Some(desc);
                    MetricState::Help(name)
                }
                (MetricState::Start, Some(Line::Type { name, kind })) => {
                    maybe_kind = Some(kind);
                    MetricState::Type(name)
                }
                (MetricState::Help(prev_name), Some(Line::Help { name, desc })) => {
                    let err_msg = match prev_name == name {
                        true => format!("Metric {prev_name} HELP section appeared multiple times"),
                        false => format!("Metric {prev_name} has no samples"),
                    };
                    let location = self.current - 1..self.current;
                    let metric_err = MetricError::new(location, err_msg);
                    self.rewind(Line::Help { name, desc });
                    return Some(Err(metric_err));
                }
                (MetricState::Help(prev_name), Some(Line::Type { name, kind })) => {
                    match prev_name == name {
                        true => {
                            maybe_kind = Some(kind);
                            MetricState::Type(name)
                        }
                        false => {
                            let location = self.current - 1..self.current;
                            let metric_err = MetricError::new(
                                location,
                                format!("Metric {prev_name} has no samples"),
                            );
                            self.rewind(Line::Type { name, kind });
                            return Some(Err(metric_err));
                        }
                    }
                }
                (MetricState::Type(prev_name), Some(Line::Type { name, kind })) => {
                    let err_msg = match prev_name == name {
                        true => format!("Metric {prev_name} TYPE section appeared multiple times"),
                        false => format!("Metric {prev_name} has no samples"),
                    };
                    let location = self.current - 1..self.current;
                    let metric_err = MetricError::new(location, err_msg);
                    self.rewind(Line::Type { name, kind });
                    return Some(Err(metric_err));
                }
                (MetricState::Type(prev_name), Some(Line::Help { name, desc })) => {
                    match prev_name == name {
                        // This is against the specs but we'll be forgiving
                        true => {
                            // leave the state as is
                            maybe_desc = Some(desc);
                            MetricState::Help(prev_name)
                        }
                        false => {
                            let location = self.current - 1..self.current;
                            let metric_err = MetricError::new(
                                location,
                                format!("Metric {prev_name} has no samples"),
                            );
                            self.rewind(Line::Help { name, desc });
                            return Some(Err(metric_err));
                        }
                    }
                }
                (MetricState::Type(prev_name) | MetricState::Help(prev_name), None) => {
                    let location = self.current - 1..self.current;
                    let metric_err =
                        MetricError::new(location, format!("Metric {prev_name} has no samples"));
                    return Some(Err(metric_err));
                }
                (
                    MetricState::Start,
                    Some(Line::Sample {
                        name,
                        labels,
                        value,
                    }),
                ) => {
                    samples.push(Sample::new(labels, value));
                    MetricState::Sample(name)
                }
                (
                    MetricState::Type(prev_name) | MetricState::Help(prev_name),
                    Some(Line::Sample {
                        name,
                        labels,
                        mut value,
                    }),
                ) => {
                    match names_are_equal(&prev_name, &name, maybe_kind) {
                        LinesStatus::Equal => {}
                        LinesStatus::NewIsSum => {
                            value.value_type = ValueType::Sum;
                        }
                        LinesStatus::NewIsCount => {
                            value.value_type = ValueType::Count;
                        }
                        LinesStatus::NotEqual => {
                            let location = self.current - 1..self.current;
                            let metric_err = MetricError::new(
                                location,
                                format!("Metric {prev_name} has no samples"),
                            );
                            self.rewind(Line::Sample {
                                name,
                                labels,
                                value,
                            });
                            return Some(Err(metric_err));
                        }
                    }
                    samples.push(Sample::new(labels, value));
                    MetricState::Sample(prev_name)
                }
                (
                    MetricState::Sample(prev_name),
                    Some(Line::Sample {
                        name,
                        labels,
                        mut value,
                    }),
                ) => {
                    match names_are_equal(&prev_name, &name, maybe_kind) {
                        LinesStatus::Equal => {}
                        LinesStatus::NewIsSum => {
                            value.value_type = ValueType::Sum;
                        }
                        LinesStatus::NewIsCount => {
                            value.value_type = ValueType::Count;
                        }
                        LinesStatus::NotEqual => {
                            // It's the start of a new metric
                            self.rewind(Line::Sample {
                                name,
                                labels,
                                value,
                            });
                            let metric = Metric::new(
                                maybe_kind.unwrap_or_default(),
                                maybe_desc,
                                prev_name,
                                samples,
                            );
                            return Some(Ok(metric));
                        }
                    }
                    samples.push(Sample::new(labels, value));
                    MetricState::Sample(prev_name)
                }
                (MetricState::Sample(prev_name), Some(Line::Help { name, desc })) => {
                    // The metric ended
                    self.rewind(Line::Help { name, desc });
                    let metric = Metric::new(
                        maybe_kind.unwrap_or_default(),
                        maybe_desc,
                        prev_name,
                        samples,
                    );
                    return Some(Ok(metric));
                }
                (MetricState::Sample(prev_name), Some(Line::Type { name, kind })) => {
                    // The metric ended
                    self.rewind(Line::Type { name, kind });
                    let metric = Metric::new(
                        maybe_kind.unwrap_or_default(),
                        maybe_desc,
                        prev_name,
                        samples,
                    );
                    return Some(Ok(metric));
                }
                // Last metric
                (MetricState::Sample(name), None) => {
                    let metric =
                        Metric::new(maybe_kind.unwrap_or_default(), maybe_desc, name, samples);
                    return Some(Ok(metric));
                }
            };
        }
    }
}

/// The status of comparing two adjacent line names
#[derive(Debug, Clone, Copy)]
enum LinesStatus {
    Equal,
    NewIsSum,
    NewIsCount,
    NotEqual,
}

/// Whether the previous and the current lines belong to the same metric
///
/// Assumption: `prev_name` is always the base name. Without the suffix "_bucket" or "_sum",
/// or "_count".
fn names_are_equal(prev_name: &str, cur_name: &str, maybe_kind: Option<Type>) -> LinesStatus {
    if prev_name == cur_name {
        return LinesStatus::Equal;
    }
    if maybe_kind == Some(Type::Histogram) && cur_name == format!("{prev_name}_bucket") {
        return LinesStatus::Equal;
    }
    if [Some(Type::Histogram), Some(Type::Summary)].contains(&maybe_kind) {
        if cur_name == format!("{prev_name}_sum") {
            return LinesStatus::NewIsSum;
        }
        if cur_name == format!("{prev_name}_count") {
            return LinesStatus::NewIsCount;
        }
    }
    LinesStatus::NotEqual
}

/// The standalone function to parse a text of a scrape into a `Vec<Metric>` and a set of errors.
/// It is used by `Scrape::parse` to parse a scrape and error on the presence of any error.
/// The stages:
/// * Parses each line in the scrape into a valid Prometheus metric line.
/// * Compose multiple lines into a metric.
pub fn parse_scrape(input: &str) -> (Vec<Metric>, Option<ScrapeParseError>) {
    let lines = match scrape_lines_parser.parse(input) {
        Ok(lines) => lines,
        Err(e) => return (Vec::new(), Some(e.into())),
    };
    let mut metrics = Vec::new();
    let mut errors = Vec::new();
    for item in MetricAssembler::new(lines) {
        match item {
            Ok(metric) => metrics.push(metric),
            Err(metric_error) => errors.push(metric_error),
        }
    }
    let maybe_error = (!errors.is_empty()).then_some(errors.into());
    (metrics, maybe_error)
}

#[cfg(test)]
mod tests {
    use super::{
        comment_line_parser,
        empty_line_parser,
        float_value_parser,
        help_line_parser,
        int_value_parser,
        label_key_value_parser,
        labels_parser,
        metric_line_parser,
        name_parser,
        new_line_or_eof_parser,
        parse_label_value,
        parse_scrape,
        rest_of_the_line_parser,
        sample_line_parser,
        scrape_lines_parser,
        type_line_parser,
        Line,
        ScrapeParseError,
    };
    use crate::{
        tests::{
            init_test_logging,
            prepare_test_data,
            EXAMPLE_01,
            NODE_EXPORTER_01,
            PROMETHEUS_01,
        },
        Float,
        Type,
        ValueType,
    };
    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use tracing::info;
    use winnow::Parser;

    #[test]
    fn test_parse_name_parser() {
        init_test_logging();

        let success_cases = [
            ("key1", "key1"),
            ("a:b:c", "a:b:c"),
            ("d33", "d33"),
            ("a_233:3:", "a_233:3:"),
        ];
        for (expr, expected) in success_cases {
            info!("Testing successful expr: '{expr}'");
            let matched = name_parser.parse(expr).unwrap();
            assert_eq!(matched, expected);
        }
        let error_cases = ["", "112_abc", "a-b", "test with space"];
        for expr in error_cases {
            info!("Testing failure expr: '{expr}'");
            assert!(name_parser.parse(expr).is_err());
        }
    }

    #[test]
    fn test_label_value_parser() {
        init_test_logging();

        let success_cases = [
            (r#""Test""#, "Test"),
            (
                r#""a string -1234567890 _:@#!""#,
                "a string -1234567890 _:@#!",
            ),
            (r#""""#, ""),
            (
                r#""Cannot find file:\n\"FILE.TXT\"""#,
                r#"Cannot find file:\n\"FILE.TXT\""#,
            ),
        ];
        for (expr, expected) in success_cases {
            info!("Testing successful expr: '{expr}'");
            let matched = parse_label_value.parse(expr).unwrap();
            assert_eq!(matched, expected);
        }
        let error_cases = ["", "\"", "\"some string"];
        for expr in error_cases {
            info!("Testing failure expr: '{expr}'");
            assert!(parse_label_value.parse(expr).is_err());
        }
    }

    #[test]
    fn test_label_key_value_parser() {
        init_test_logging();

        let success_cases = [
            (r#"key1="Test""#, ("key1", "Test")),
            (r#"key1  = "Test""#, ("key1", "Test")),
            (r#"key1="Test""#, ("key1", "Test")),
            (r#"key1="""#, ("key1", "")),
            (r#"k:_e="@!2334+~`""#, ("k:_e", "@!2334+~`")),
        ];
        for (expr, (key, val)) in success_cases {
            info!("Testing successful expr: '{expr}'");
            let (recv_key, recv_val) = label_key_value_parser.parse(expr).unwrap();
            assert_eq!(key, recv_key);
            assert_eq!(val, recv_val);
        }
        let error_cases = [
            "",
            r#"key1="Test"#,
            r#""key1"="Test""#,
            "key1=",
            r#"key1 "Test""#,
        ];
        for expr in error_cases {
            info!("Testing failure expr: '{expr}'");
            assert!(label_key_value_parser.parse(expr).is_err());
        }
    }

    #[test]
    fn test_labels_parser() {
        init_test_logging();

        let success_cases = [
            r#"{key1="value1",key2="value2"}"#,
            r#"{key1="value1", key2 = "value2"}"#,
            r#"{ key1="value1",    key2 = "value2" }"#,
            r#"{ key1  =  "value1",    key2 = "value2" }"#,
        ];
        for expr in success_cases {
            info!("Testing successful expr: '{expr}'");
            let labels = labels_parser.parse(expr).unwrap();
            assert_eq!(labels.len(), 2);
            let mut iter = labels.iter();
            let label = iter.next().unwrap();
            assert_eq!("key1", label.key);
            assert_eq!("value1", label.value);
            let label = iter.next().unwrap();
            assert_eq!("key2", label.key);
            assert_eq!("value2", label.value);
        }

        let error_cases = ["", "{}", r#"{key1="value1",key2="value2""#];
        for expr in error_cases {
            info!("Testing failure expr: '{expr}'");
            assert!(label_key_value_parser.parse(expr).is_err());
        }
    }

    #[test]
    fn test_int_value_parser() {
        init_test_logging();

        let success_cases = [
            ("0", 0),
            ("1", 1),
            ("-1", -1),
            ("100000", 100000),
            ("-1345555", -1345555),
        ];
        for (expr, val) in success_cases {
            info!("Testing successful expr: '{expr}'");
            let recv_val = int_value_parser.parse(expr).unwrap();
            assert_eq!(val, recv_val);
        }

        let error_cases = ["", "b123"];
        for expr in error_cases {
            info!("Testing failure expr: '{expr}'");
            assert!(int_value_parser.parse(expr).is_err());
        }
    }

    #[test]
    fn test_float_value_parser() {
        init_test_logging();

        let success_cases = [
            ("0", 0.0),
            ("0.0", 0.0),
            ("1.0", 1.0),
            ("-1.0", -1.0),
            ("Inf", f64::INFINITY),
            ("+Inf", f64::INFINITY),
            ("-Inf", f64::NEG_INFINITY),
            ("1e4", 1.0e4),
            ("NaN", f64::NAN),
            ("nan", f64::NAN),
            ("NAN", f64::NAN),
            ("-1.23e+1", -1.23e+1),
            ("-1.23e-1", -1.23e-1),
            ("+.22", 0.22),
            (".33", 0.33),
        ];
        for (expr, num) in success_cases {
            info!("Testing successful expr: '{expr}'");
            let recv_val = float_value_parser.parse(expr).unwrap();
            assert_eq!(expr, *recv_val);
            let parsed = recv_val.as_f64();
            assert!(parsed == num || (parsed.is_nan() && num.is_nan()));
        }
    }

    #[test]
    fn test_new_line_or_eof_parser() {
        init_test_logging();

        let success_cases = ["", "\n"];
        for expr in success_cases {
            info!("Testing successful expr: '{expr}'");
            let res = new_line_or_eof_parser.parse(expr);
            assert_eq!(res, Ok(()));
        }

        let error_cases = [" ", "\t", "abc"];
        for expr in error_cases {
            info!("Testing failure expr: '{expr}'");
            let res = new_line_or_eof_parser.parse(expr);
            assert!(res.is_err());
        }
    }

    #[test]
    fn test_rest_of_the_line_parser() {
        init_test_logging();

        let success_cases = [("1\n", "1"), ("   1\n", "1")];
        for (expr, expected) in success_cases {
            info!("Testing successful expr: '{expr}'");
            let parsed = rest_of_the_line_parser.parse(expr).unwrap();
            assert_eq!(parsed, expected);
        }

        let error_cases = [""];
        for expr in error_cases {
            info!("Testing failure expr: '{expr}'");
            let res = rest_of_the_line_parser.parse(expr);
            assert!(res.is_err());
        }
    }

    #[test]
    fn test_empty_line_parser() {
        init_test_logging();

        let success_cases = ["\n", "   \n", "\t\n"];
        for expr in success_cases {
            info!("Testing successful expr: '{expr}'");
            let res = empty_line_parser.parse(expr);
            assert!(res.is_ok());
        }

        let error_cases = ["", "not-empty\n", "@\n", "     "];
        for expr in error_cases {
            info!("Testing failure expr: '{expr}'");
            let res = empty_line_parser.parse(expr);
            assert!(res.is_err());
        }
    }

    #[test]
    fn test_sample_line_parser_01() {
        init_test_logging();

        let cases = [
            r#"data_sent:bytes{th_id="worker_0",type="duplex"} 1395 -1"#,
            "data_sent:bytes{th_id=\"worker_0\",type=\"duplex\"} 1395 -1\n",
            "data_sent:bytes{th_id=\"worker_0\",type=\"duplex\"} 1395 -1   \n",
            r#"data_sent:bytes { th_id = "worker_0" , type = "duplex" }   1395  -1  "#,
        ];
        for expr in cases {
            info!("Testing successful expr: '{expr}'");
            let (name, labels, value) = match sample_line_parser.parse(expr) {
                Ok(Line::Sample {
                    name,
                    labels,
                    value,
                }) => (name, labels, value),
                res => panic!("Received unexpected {res:?}"),
            };
            assert_eq!(name, "data_sent:bytes");
            assert_eq!(labels.len(), 2);
            let mut iter = labels.iter();
            let label = iter.next().unwrap();
            assert_eq!("th_id", label.key);
            assert_eq!("worker_0", label.value);
            let label = iter.next().unwrap();
            assert_eq!("type", label.key);
            assert_eq!("duplex", label.value);
            assert_eq!(*value.value, "1395");
            assert_eq!(value.timestamp, Some(-1));
        }
    }

    #[test]
    fn test_sample_line_parser_failure_01() {
        init_test_logging();

        let cases = [
            r#"data_sent:bytes{th_id="worker_0",type="duplex"}"#,
            r#"data_sent:bytes { th_id = "worker_0" , type = "duplex" }   1395  -1  some-more-text"#,
        ];
        for expr in cases {
            info!("Testing failure expr: '{expr}'");
            assert!(sample_line_parser.parse(expr).is_err());
        }
    }

    #[test]
    fn test_comment_line_parser() {
        init_test_logging();

        let success_cases = [
            ("# a comment", "a comment"),
            ("  #    Something else", "Something else"),
        ];
        for (expr, expected_comment) in success_cases {
            info!("Testing successful expr: '{expr}'");
            let comment = match comment_line_parser.parse(expr) {
                Ok(Line::Comment(comment)) => comment,
                res => panic!("Received unexpected {res:?}"),
            };
            assert_eq!(expected_comment, comment);
        }

        let error_cases = ["", "^# something"];
        for expr in error_cases {
            info!("Testing failure expr: '{expr}'");
            assert!(comment_line_parser.parse(expr).is_err());
        }
    }

    #[test]
    fn test_help_line_parser() {
        init_test_logging();

        let success_cases = [
            (
                "# HELP http_request_duration_seconds A histogram of the request duration.",
                (
                    "http_request_duration_seconds",
                    "A histogram of the request duration.",
                ),
            ),
            (
                "  # HELP name long description",
                ("name", "long description"),
            ),
        ];
        for (expr, (expected_name, expected_desc)) in success_cases {
            info!("Testing successful expr: '{expr}'");
            let (name, desc) = match help_line_parser.parse(expr) {
                Ok(Line::Help { name, desc }) => (name, desc),
                res => panic!("Received unexpected {res:?}"),
            };
            assert_eq!(expected_name, name);
            assert_eq!(expected_desc, desc);
        }

        let error_cases = ["", "# something", "# HELP"];
        for expr in error_cases {
            info!("Testing failure expr: '{expr}'");
            assert!(help_line_parser.parse(expr).is_err());
        }
    }

    #[test]
    fn test_type_line_parser() {
        init_test_logging();

        let expected_name = "test";
        let success_cases = [
            ("# TYPE test histogram", Type::Histogram),
            ("# TYPE test summary", Type::Summary),
            ("# TYPE test counter", Type::Counter),
            ("# TYPE test gauge", Type::Gauge),
            ("# TYPE test untyped", Type::Untyped),
            ("# TYPE test Summary", Type::Summary),
        ];
        for (expr, expected_kind) in success_cases {
            info!("Testing successful expr: '{expr}'");
            let (name, kind) = match type_line_parser.parse(expr) {
                Ok(Line::Type { name, kind }) => (name, kind),
                res => panic!("Received unexpected {res:?}"),
            };
            assert_eq!(expected_name, name);
            assert_eq!(expected_kind, kind);
        }

        let error_cases = ["", "# TYPE test something", "# TYPE"];
        for expr in error_cases {
            info!("Testing failure expr: '{expr}'");
            assert!(help_line_parser.parse(expr).is_err());
        }
    }

    #[test]
    fn test_metric_line_parser() {
        init_test_logging();

        let cases = [
            "# TYPE test histogram",
            "# TYPE test summary",
            "# TYPE test counter",
            "# TYPE test gauge",
            "# TYPE test untyped",
            "# TYPE test Summary",
            "# HELP http_request_duration_seconds A histogram of the request duration.",
            "  # HELP name long description",
            "# a comment",
            r#"data_sent:bytes{th_id="worker_0",type="duplex"} 1395 -1"#,
            r#"tower:histogram_bucket{name="handler",th_id="worker_0",type="1",le="64"} 0"#,
            r#"tower:histogram_bucket{name="handler",th_id="worker_0",type="1",le="+Inf"} 0"#,
            r#"tower:histogram_sum{name="handler",th_id="worker_0",type="1"} 0"#,
            r#"tower:histogram_count{name="handler",th_id="worker_0",type="1"} 0"#,
        ];
        for expr in cases {
            info!("Testing successful expr: '{expr}'");
            assert!(metric_line_parser.parse(expr).is_ok());
        }
    }

    #[rstest]
    fn test_scrape_parser(#[values(EXAMPLE_01, NODE_EXPORTER_01, PROMETHEUS_01)] data: &str) {
        init_test_logging();

        let expected_len = data.lines().count();
        let lines = match scrape_lines_parser.parse(data) {
            Ok(lines) => lines,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(lines.len(), expected_len);
    }

    #[test]
    fn test_scrape_success_01() {
        init_test_logging();

        let input = r#"
                # TYPE go_memstats_frees_total counter
                # HELP go_memstats_frees_total Total number of frees.
                go_memstats_frees_total 4.130418363e+09
            "#;
        let input = prepare_test_data(input);
        let (mut metrics, maybe_error) = parse_scrape(&input);
        assert!(maybe_error.is_none());
        assert_eq!(metrics.len(), 1);
        let metric = metrics.pop().unwrap();
        assert_eq!(metric.kind, Type::Counter);
        assert_eq!(metric.help_desc.as_deref(), Some("Total number of frees."));
        assert_eq!(metric.name, "go_memstats_frees_total");
        assert_eq!(
            metric.samples[0].value.value,
            Float("4.130418363e+09".into())
        );
    }

    #[test]
    fn test_scrape_parse_failure_01() {
        init_test_logging();

        let inputs = [
            r#"
                # HELP http_request_duration_seconds A histogram of the request duration.
                # TYPE http_request_duration_seconds histogram
            "#,
            r#"
                # TYPE http_request_duration_seconds histogram
            "#,
        ];
        for input in inputs.iter().map(|i| prepare_test_data(i)) {
            let (metrics, maybe_error) = parse_scrape(&input);
            assert!(metrics.is_empty());
            let ScrapeParseError::Collect(metric_errors) = maybe_error.unwrap() else {
                panic!("expected metric errors");
            };
            assert_eq!(metric_errors.len(), 1);
            assert_eq!(
                metric_errors[0].reason,
                "Metric http_request_duration_seconds has no samples"
            );
        }
    }

    #[test]
    fn test_scrape_parse_failure_02() {
        init_test_logging();

        let inputs = [
            r#"
                # TYPE http_request_duration_seconds histogram
                # TYPE http_request_duration_seconds histogram
            "#,
            r#"
                # HELP go_info Information about the Go environment.
                # HELP go_info Information about the Go environment.
            "#,
        ];
        let error_reasons = [
            "Metric http_request_duration_seconds TYPE section appeared multiple times",
            "Metric go_info HELP section appeared multiple times",
        ];
        for (input, reason) in inputs
            .iter()
            .map(|i| prepare_test_data(i))
            .zip(error_reasons)
        {
            let (metrics, maybe_error) = parse_scrape(&input);
            assert!(metrics.is_empty());
            let ScrapeParseError::Collect(metric_errors) = maybe_error.unwrap() else {
                panic!("expected metric errors");
            };
            assert_eq!(metric_errors.len(), 2);
            assert_eq!(metric_errors[0].reason, reason);
        }
    }

    #[test]
    fn test_scrape_parse_failure_03() {
        init_test_logging();

        let inputs = [
            r#"
                # TYPE http_request_duration_seconds histogram
                # TYPE http_request counter
            "#,
            r#"
                # HELP go_info Information about the Go environment.
                # HELP not_go_info No information about the Go environment.
            "#,
        ];
        let error_reasons = [
            (
                "Metric http_request_duration_seconds has no samples",
                "Metric http_request has no samples",
            ),
            (
                "Metric go_info has no samples",
                "Metric not_go_info has no samples",
            ),
        ];
        for (input, reasons) in inputs
            .iter()
            .map(|i| prepare_test_data(i))
            .zip(error_reasons)
        {
            let (metrics, maybe_error) = parse_scrape(&input);
            assert!(metrics.is_empty());
            let ScrapeParseError::Collect(metric_errors) = maybe_error.unwrap() else {
                panic!("expected metric errors");
            };
            assert_eq!(metric_errors.len(), 2);
            assert_eq!(metric_errors[0].reason, reasons.0);
            assert_eq!(metric_errors[1].reason, reasons.1);
        }
    }

    #[test]
    fn test_scrape_parse_failure_04() {
        init_test_logging();

        let input = r#"
            # HELP go_info Information about the Go environment.
            # TYPE http_request counter
        "#;
        let input = prepare_test_data(input);
        let (metrics, maybe_error) = parse_scrape(&input);
        assert!(metrics.is_empty());
        let ScrapeParseError::Collect(metric_errors) = maybe_error.unwrap() else {
            panic!("expected metric errors");
        };
        assert_eq!(metric_errors.len(), 2);
        assert_eq!(metric_errors[0].reason, "Metric go_info has no samples");
        assert_eq!(
            metric_errors[1].reason,
            "Metric http_request has no samples"
        );
    }

    #[test]
    fn test_scrape_parse_failure_05() {
        init_test_logging();

        let input = r#"
            # TYPE http_request counter
            # HELP go_info Information about the Go environment.
        "#;
        let input = prepare_test_data(input);
        let (metrics, maybe_error) = parse_scrape(&input);
        assert!(metrics.is_empty());
        let ScrapeParseError::Collect(metric_errors) = maybe_error.unwrap() else {
            panic!("expected metric errors");
        };
        assert_eq!(metric_errors.len(), 2);
        assert_eq!(
            metric_errors[0].reason,
            "Metric http_request has no samples"
        );
        assert_eq!(metric_errors[1].reason, "Metric go_info has no samples");
    }

    #[test]
    fn test_scrape_parse_mix_01() {
        init_test_logging();

        let input = r#"
                # HELP go_info Information about the Go environment.
                go_info{version="go1.15.6"} 1
                # TYPE go_memstats_alloc_bytes gauge
            "#;
        let input = prepare_test_data(input);

        let (mut metrics, maybe_error) = parse_scrape(&input);
        assert_eq!(metrics.len(), 1);
        let mut metric = metrics.pop().unwrap();
        assert_eq!(metric.name, "go_info");
        assert_eq!(
            metric.help_desc.as_deref(),
            Some("Information about the Go environment.")
        );
        assert_eq!(metric.samples.len(), 1);
        let sample = metric.samples.pop().unwrap();
        assert_eq!(sample.value.value_type, ValueType::Sample);
        assert_eq!(sample.value.value, Float("1".into()));
        assert_eq!(sample.value.timestamp, None);
        assert_eq!(sample.labels.len(), 1);
        assert_eq!(sample.labels[0].key, "version");
        assert_eq!(sample.labels[0].value, "go1.15.6");
        let ScrapeParseError::Collect(metric_errors) = maybe_error.unwrap() else {
            panic!("expected metric errors");
        };
        assert_eq!(metric_errors.len(), 1);
        assert_eq!(
            metric_errors[0].reason,
            "Metric go_memstats_alloc_bytes has no samples"
        );
    }
}
