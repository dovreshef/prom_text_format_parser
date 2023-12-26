use prom_text_format_parser::Scrape;

fn main() {
    let path = std::env::args().nth(1).expect("A path to scrape text");
    let text = std::fs::read_to_string(path).expect("file read");
    let mut scrape = Scrape::parse(&text).expect("valid scrape");

    // Add a label to all metrics
    scrape.add_label("source", "invalid");

    // Print the scrape in the Prometheus exposition text format
    println!("{scrape}");
}
