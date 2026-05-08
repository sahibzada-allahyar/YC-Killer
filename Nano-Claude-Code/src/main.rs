use nano_claude_code::cli;

fn main() {
    let exit_code = match cli::run(std::env::args().collect()) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("{err}");
            err.exit_code()
        }
    };

    std::process::exit(exit_code);
}
