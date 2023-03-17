use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Where to load the model path from
    #[arg(long, short = 'm')]
    pub model_path: String,

    /// Sets the number of threads to use
    #[arg(long, short = 't', default_value_t = std::thread::available_parallelism().unwrap().get())]
    pub num_threads: usize,

    /// Sets the size of the context (in tokens). Allows feeding longer prompts.
    /// Note that this affects memory. TODO: Unsure how large the limit is.
    #[arg(long, default_value_t = 127)]
    pub num_ctx_tokens: usize,
}
