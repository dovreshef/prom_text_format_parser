#![forbid(unsafe_code)]
//! The Prometheus exposition format is taken from here:
//! <https://prometheus.io/docs/instrumenting/exposition_formats/>
use derive_more::Constructor;
pub use parser::{
    parse_scrape,
    MetricError,
    ScrapeParseError,
};
use std::fmt::Display;

mod parser;

/// The possible types of Prometheus metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, strum::EnumString, strum::Display)]
#[strum(ascii_case_insensitive)]
#[strum(serialize_all = "snake_case")]
pub enum Type {
    Counter,
    Gauge,
    #[default]
    Untyped,
    Summary,
    Histogram,
}

/// A single label in a sample.
///
/// Example:
/// ```text
/// name="a"
/// ```
#[derive(Debug, Clone, PartialEq, Eq, derive_more::Constructor)]
pub struct Label {
    /// Label key
    pub key: String,
    /// Label value (without the quotes)
    pub value: String,
}

/// A set of labels identifying a sample.
///
/// Example:
/// ```text
/// {name="a",id="1",type="x"}
/// ```
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Default,
    derive_more::Deref,
    derive_more::DerefMut,
    derive_more::From,
)]
#[repr(transparent)]
pub struct Labels(Vec<Label>);

impl Display for Labels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            return Ok(());
        }
        let last_idx = self.0.len() - 1;
        f.write_str("{")?;
        for (idx, label) in self.0.iter().enumerate() {
            f.write_str(&label.key)?;
            f.write_str("=\"")?;
            f.write_str(&label.value)?;
            f.write_str("\"")?;
            if idx != last_idx {
                f.write_str(",")?;
            }
        }
        f.write_str("}")?;
        Ok(())
    }
}

/// The possible types of a `Value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Sample,
    // The following two are only relevant for Summary, Histogram
    Sum,
    Count,
}

/// Float is a float as represented by Go's ParseFloat() function.
/// In addition to standard numerical values, NaN, +Inf, and -Inf are valid values representing
/// not a number, positive infinity, and negative infinity, respectively.
/// Since it's very hard to render float as Go renders them in Rust, and we don't need the
/// actual value - we keep the value as String, and only validate on parsing that it's a
/// valid float.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    derive_more::Deref,
    derive_more::DerefMut,
    derive_more::From,
    derive_more::FromStr,
)]
#[repr(transparent)]
pub struct Float(String);

impl Float {
    /// Parse and return the value as an `f64`
    pub fn as_f64(&self) -> f64 {
        self.0.parse().unwrap()
    }
}

/// A Prometheus metric value, for a specific combination of labels.
#[derive(Debug, Clone, PartialEq, Eq, Constructor)]
pub struct Value {
    /// Whether this is a regular sample value, or a Sum or a Count value.
    pub value_type: ValueType,
    /// The sample value, still a string, but validated to be a float.
    pub value: Float,
    /// The timestamp is an int64 (milliseconds since epoch, i.e. 1970-01-01 00:00:00 UTC,
    /// excluding leap seconds), represented as required by Go's ParseInt() function.
    pub timestamp: Option<i64>,
}

impl Display for Value {
    /// Print the the value, followed by an optional timestamp
    /// Example:
    /// ```text
    /// 3 1395066363000
    /// ```
    ///
    /// NOTES:
    /// * New line is not added.
    /// * The values are assumed to be escaped already (i.e. '\\', '\\n', '\\"').
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.value.as_str())?;
        if let Some(ts) = self.timestamp {
            write!(f, " {ts}")?;
        }
        Ok(())
    }
}

/// A representation of a sample line, containing the labels and the value.
#[derive(Debug, Clone, PartialEq, Eq, Constructor)]
pub struct Sample {
    pub labels: Labels,
    pub value: Value,
}

/// A metric.
///
/// An example:
/// ```text
/// # this does not include health checks
/// # HELP http_requests_total The total number of HTTP requests.
/// # TYPE http_requests_total counter
/// http_requests_total{method="post",code="200"} 1027 1395066363000
/// http_requests_total{method="post",code="400"}    3 1395066363000
/// ```
/// Or
/// ```text
/// # Finally a summary, which has a complex representation, too:
/// # HELP rpc_duration_seconds A summary of the RPC duration in seconds.
/// # TYPE rpc_duration_seconds summary
/// rpc_duration_seconds{quantile="0.01"} 3102
/// rpc_duration_seconds{quantile="0.05"} 3272
/// rpc_duration_seconds{quantile="0.5"} 4773
/// rpc_duration_seconds{quantile="0.9"} 9001
/// rpc_duration_seconds{quantile="0.99"} 76656
/// rpc_duration_seconds_sum 1.7560473e+07
/// rpc_duration_seconds_count 2693
/// ```
#[derive(Debug, Clone, Constructor)]
pub struct Metric {
    pub kind: Type,
    /// A comment line above the metric which does start with HELP
    pub help_desc: Option<String>,
    /// The name of the metric, excluding the labels.
    pub name: String,
    /// The data
    pub samples: Vec<Sample>,
}

impl Metric {
    /// Add a label to all the samples in the metric
    pub fn add_label(&mut self, key: &str, value: &str) {
        for sample in &mut self.samples {
            sample.labels.0.push(Label::new(key.into(), value.into()))
        }
    }

    /// remove a label from all the samples in the metric, if it exists
    pub fn remove_label(&mut self, key: &str, value: &str) {
        for sample in &mut self.samples {
            let maybe_idx = sample
                .labels
                .0
                .iter()
                .position(|label| label.key == key && label.value == value);
            if let Some(idx) = maybe_idx {
                sample.labels.0.remove(idx);
            }
        }
    }
}

impl Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(help) = self.help_desc.as_deref() {
            writeln!(f, "# HELP {} {help}", self.name)?;
        }
        if self.kind != Type::Untyped {
            writeln!(f, "# TYPE {} {}", self.name, self.kind)?;
        }
        for sample in self.samples.iter() {
            let suffix = match (self.kind, sample.value.value_type) {
                (Type::Histogram, ValueType::Sample) => "_bucket",
                (Type::Histogram | Type::Summary, ValueType::Sum) => "_sum",
                (Type::Histogram | Type::Summary, ValueType::Count) => "_count",
                _ => "",
            };
            writeln!(f, "{}{suffix}{} {}", self.name, sample.labels, sample.value)?;
        }
        Ok(())
    }
}

/// A single scrape.
/// Parses a textual scrape into a vector of metrics.
/// Implements `Display` to print the metrics in the Prometheus exposition text format.
///
/// NOTES:
/// The parsing is not lossless. Comments (excluding TYPE, HELP comments) and empty lines
/// are discarded.
#[derive(Debug, Clone)]
pub struct Scrape {
    /// The metrics given in the scrape.
    pub metrics: Vec<Metric>,
}

impl Display for Scrape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for metric in self.metrics.iter() {
            write!(f, "{metric}")?;
        }
        Ok(())
    }
}

impl Scrape {
    pub fn parse(data: &str) -> Result<Self, ScrapeParseError> {
        let (metrics, maybe_error) = parse_scrape(data);
        match maybe_error {
            Some(error) => Err(error),
            None => Ok(Self { metrics }),
        }
    }

    /// Add a label to all the metrics in the scrape
    pub fn add_label(&mut self, key: &str, value: &str) {
        for metric in &mut self.metrics {
            metric.add_label(key, value);
        }
    }

    /// Remove a label from all the metrics in the scrape, if it exists
    pub fn remove_label(&mut self, key: &str, value: &str) {
        for metric in &mut self.metrics {
            metric.remove_label(key, value);
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::{
        Scrape,
        Type,
        ValueType,
    };
    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use std::{
        str::FromStr,
        sync::Once,
    };
    use tracing_subscriber::EnvFilter;

    static INIT_LOGGER: Once = Once::new();

    pub(crate) fn init_test_logging() {
        INIT_LOGGER.call_once(|| {
            tracing_subscriber::fmt::fmt()
                .with_env_filter(EnvFilter::new("warn,prom_text_format_parser=debug"))
                .init();
        });
    }

    pub const EXAMPLE_01: &str = include_str!("../test_data/example-01.txt");
    pub const NODE_EXPORTER_01: &str = include_str!("../test_data/node-exporter-01.txt");
    pub const PROMETHEUS_01: &str = include_str!("../test_data/prometheus-01.txt");
    pub const FAIL_01: &str = include_str!("../test_data/fail-no-sample-01.txt");

    #[test]
    fn test_value_type_conversion() {
        let cases = [
            ("untyped", Type::Untyped),
            ("UNTYPED", Type::Untyped),
            ("counter", Type::Counter),
            ("COUNTER", Type::Counter),
            ("gauge", Type::Gauge),
            ("GAUGE", Type::Gauge),
            ("histogram", Type::Histogram),
            ("HISTOGRAM", Type::Histogram),
            ("summary", Type::Summary),
            ("SUMMARY", Type::Summary),
        ];
        for (expr, expected) in cases {
            let found = Type::from_str(expr).unwrap();
            assert_eq!(found, expected);
        }
    }

    #[rstest]
    fn test_end_to_end(#[values(NODE_EXPORTER_01, PROMETHEUS_01)] data: &str) {
        init_test_logging();

        let scrape = Scrape::parse(data).unwrap();
        let printed = format!("{scrape}");
        assert_eq!(data, printed);
    }

    #[rstest]
    fn test_scrape_failure(#[values("", FAIL_01)] data: &str) {
        init_test_logging();

        let res = Scrape::parse(data);
        assert!(res.is_err());
    }

    #[test]
    fn test_complex_scrape() {
        init_test_logging();

        let data = EXAMPLE_01;
        let metrics = Scrape::parse(data).unwrap().metrics;
        assert_eq!(metrics.len(), 6);

        let metric1 = metrics[0].clone();
        assert_eq!(metric1.name, "http_requests_total");
        assert_eq!(
            metric1.help_desc.as_deref(),
            Some("The total number of HTTP requests.")
        );
        assert_eq!(metric1.kind, Type::Counter);
        assert_eq!(metric1.samples.len(), 2);
        assert_eq!(metric1.samples[0].labels.len(), 2);
        assert_eq!(metric1.samples[0].labels[0].key, "method");
        assert_eq!(metric1.samples[0].labels[0].value, "post");
        assert_eq!(metric1.samples[0].labels[1].key, "code");
        assert_eq!(metric1.samples[0].labels[1].value, "200");
        assert_eq!(*metric1.samples[0].value.value, "1027");
        assert_eq!(metric1.samples[0].value.value_type, ValueType::Sample);
        assert_eq!(metric1.samples[0].value.timestamp, Some(1395066363000));
        assert_eq!(metric1.samples[1].labels.len(), 2);
        assert_eq!(metric1.samples[1].labels[0].key, "method");
        assert_eq!(metric1.samples[1].labels[0].value, "post");
        assert_eq!(metric1.samples[1].labels[1].key, "code");
        assert_eq!(metric1.samples[1].labels[1].value, "400");
        assert_eq!(*metric1.samples[1].value.value, "3");
        assert_eq!(metric1.samples[1].value.value_type, ValueType::Sample);
        assert_eq!(metric1.samples[1].value.timestamp, Some(1395066363000));

        let metric2 = metrics[1].clone();
        assert_eq!(metric2.name, "msdos_file_access_time_seconds");
        assert_eq!(metric2.help_desc.as_deref(), None);
        assert_eq!(metric2.kind, Type::Untyped);
        assert_eq!(metric2.samples.len(), 1);
        assert_eq!(metric2.samples[0].labels.len(), 2);
        assert_eq!(metric2.samples[0].labels[0].key, "path");
        assert_eq!(metric2.samples[0].labels[0].value, r"C:\\DIR\\FILE.TXT");
        assert_eq!(metric2.samples[0].labels[1].key, "error");
        assert_eq!(
            metric2.samples[0].labels[1].value,
            r#"Cannot find file:\n\"FILE.TXT\""#
        );
        assert_eq!(*metric2.samples[0].value.value, "1.458255915e9");
        assert_eq!(metric2.samples[0].value.value_type, ValueType::Sample);
        assert_eq!(metric2.samples[0].value.timestamp, None);

        let metric3 = metrics[2].clone();
        assert_eq!(metric3.name, "metric_without_timestamp_and_labels");
        assert_eq!(metric3.help_desc.as_deref(), None);
        assert_eq!(metric3.kind, Type::Untyped);
        assert_eq!(metric3.samples.len(), 1);
        assert_eq!(metric3.samples[0].labels.len(), 0);
        assert_eq!(*metric3.samples[0].value.value, "12.47");
        assert_eq!(metric3.samples[0].value.value_type, ValueType::Sample);
        assert_eq!(metric3.samples[0].value.timestamp, None);

        let metric4 = metrics[3].clone();
        assert_eq!(metric4.name, "something_weird");
        assert_eq!(metric4.help_desc.as_deref(), None);
        assert_eq!(metric4.kind, Type::Untyped);
        assert_eq!(metric4.samples.len(), 1);
        assert_eq!(metric4.samples[0].labels.len(), 1);
        assert_eq!(metric4.samples[0].labels[0].key, "problem");
        assert_eq!(metric4.samples[0].labels[0].value, "division by zero");
        assert_eq!(*metric4.samples[0].value.value, "+Inf");
        assert_eq!(metric4.samples[0].value.value_type, ValueType::Sample);
        assert_eq!(metric4.samples[0].value.timestamp, Some(-3982045));
    }

    // Remove starting & trailing space from all lines.
    // Remove empty lines.
    fn prepare_test_data(data: &str) -> String {
        data.lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn test_add_label() {
        let input = r#"
            # HELP http_request_duration_seconds A histogram of the request duration.
            # TYPE http_request_duration_seconds histogram
            http_request_duration_seconds_bucket{le="0.05"} 24054
            http_request_duration_seconds_bucket{le="0.1"} 33444
            http_request_duration_seconds_bucket{le="0.2"} 100392
            http_request_duration_seconds_bucket{le="0.5"} 129389
            http_request_duration_seconds_bucket{le="1"} 133988
            http_request_duration_seconds_bucket{le="+Inf"} 144320
            http_request_duration_seconds_sum 53423
            http_request_duration_seconds_count 144320  
        "#;
        let expected = r#"
            # HELP http_request_duration_seconds A histogram of the request duration.
            # TYPE http_request_duration_seconds histogram
            http_request_duration_seconds_bucket{le="0.05",one="two",three="3"} 24054
            http_request_duration_seconds_bucket{le="0.1",one="two",three="3"} 33444
            http_request_duration_seconds_bucket{le="0.2",one="two",three="3"} 100392
            http_request_duration_seconds_bucket{le="0.5",one="two",three="3"} 129389
            http_request_duration_seconds_bucket{le="1",one="two",three="3"} 133988
            http_request_duration_seconds_bucket{le="+Inf",one="two",three="3"} 144320
            http_request_duration_seconds_sum{one="two",three="3"} 53423
            http_request_duration_seconds_count{one="two",three="3"} 144320  
        "#;
        let input = prepare_test_data(input);
        let mut expected = prepare_test_data(expected);
        // We write an empty line at the end, per the specs
        expected.push('\n');
        let mut scrape = Scrape::parse(&input).unwrap();
        scrape.add_label("one", "two");
        scrape.add_label("three", "3");
        let output = format!("{scrape}");
        assert_eq!(output, expected);
    }

    #[test]
    fn test_remove_label() {
        let input = r#"
            # HELP http_request_duration_seconds A histogram of the request duration.
            # TYPE http_request_duration_seconds histogram
            http_request_duration_seconds_bucket{le="0.05",one="two",three="3"} 24054
            http_request_duration_seconds_bucket{le="0.1",one="two",three="3"} 33444
            http_request_duration_seconds_bucket{le="0.2",one="two",three="3"} 100392
            http_request_duration_seconds_bucket{le="0.5",one="two",three="3"} 129389
            http_request_duration_seconds_bucket{le="1",one="two",three="3"} 133988
            http_request_duration_seconds_bucket{le="+Inf",one="two",three="3"} 144320
            http_request_duration_seconds_sum{one="two",three="3"} 53423
            http_request_duration_seconds_count{one="two",three="3"} 144320
        "#;
        let expected = r#"
            # HELP http_request_duration_seconds A histogram of the request duration.
            # TYPE http_request_duration_seconds histogram
            http_request_duration_seconds_bucket{le="0.05"} 24054
            http_request_duration_seconds_bucket{le="0.1"} 33444
            http_request_duration_seconds_bucket{le="0.2"} 100392
            http_request_duration_seconds_bucket{le="0.5"} 129389
            http_request_duration_seconds_bucket{le="1"} 133988
            http_request_duration_seconds_bucket{le="+Inf"} 144320
            http_request_duration_seconds_sum 53423
            http_request_duration_seconds_count 144320
        "#;
        let input = prepare_test_data(input);
        let mut expected = prepare_test_data(expected);
        // We write an empty line at the end, per the specs
        expected.push('\n');
        let mut scrape = Scrape::parse(&input).unwrap();
        scrape.remove_label("one", "two");
        scrape.remove_label("three", "3");
        let output = format!("{scrape}");
        assert_eq!(output, expected);
    }
}
