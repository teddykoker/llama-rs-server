use clap::Parser;

use axum::{extract::State, routing::post, Json, Router};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

mod args;
mod utils;

#[tokio::main]
async fn main() {
    let args = args::Args::parse();

    let (sender, receiver) = flume::unbounded::<ThreadGenerateRequest>();

    utils::start_model_thread(
        receiver,
        args.model_path,
        args.num_threads.try_into().unwrap(),
        args.num_ctx_tokens.try_into().unwrap(),
    );

    let app = Router::new()
        .route("/completions", post(completions))
        .with_state(sender);

    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}

enum Token {
    Token(String),
    EndOfText,
}

pub struct ThreadGenerateRequest(GenerateRequest, flume::Sender<Token>);

#[derive(Deserialize)]
#[serde(default)]
struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub batch_size: usize,
    pub repeat_last_n: usize,
    pub repeat_penalty: f32,
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
}
impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt: Default::default(),
            max_tokens: 128,
            batch_size: 8,
            repeat_last_n: 64,
            repeat_penalty: 1.30,
            temp: 0.80,
            top_k: 40,
            top_p: 0.95,
        }
    }
}

#[derive(Serialize)]
struct GenerateResponse {
    text: String,
}

async fn completions(
    State(sender): State<flume::Sender<ThreadGenerateRequest>>,
    Json(request): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    let (token_sender, token_receiver) = flume::unbounded();

    sender
        .send(ThreadGenerateRequest(request, token_sender))
        .unwrap();

    let mut text = String::new();
    let mut stream = token_receiver.into_stream();

    while let Some(token) = stream.next().await {
        match token {
            Token::Token(t) => {
                text += t.as_str();
            }
            Token::EndOfText => {}
        }
    }

    Json(GenerateResponse { text })
}
