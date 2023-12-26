![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/dovreshef/prom_text_format_parser/.github%2Fworkflows%2Frust.yml)
![docs.rs](https://img.shields.io/docsrs/prom_text_format_parser)
![Crates.io](https://img.shields.io/crates/l/prom_text_format_parser)
![Codecov](https://img.shields.io/codecov/c/github/dovreshef/prom_text_format_parser)
![Crates.io](https://img.shields.io/crates/v/prom_text_format_parser)

This is a parser and printer for the Prometheus exposition text format.

See [here](https://prometheus.io/docs/instrumenting/exposition_formats/) for a detailed description
of the format.

Usage example:

```rust
    let path = std::env::args().nth(1).expect("A path to scrape text");
    let text = std::fs::read_to_string(path).expect("file read");
    let mut scrape = Scrape::parse(&text).expect("valid scrape");

    // Add a label to all metrics
    scrape.add_label("source", "invalid");
    // Remove a label from all metrics
    scrape.remove_label("source", "invalid");

    // format the scrape in the Prometheus exposition text format
    let rendered = format!("{scrape}");
    assert_eq!(text, rendered);

```
