use axum::{
    http::StatusCode,
    response::IntoResponse,
    routing::{post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

#[tokio::main]
async fn main() {
    // initialize tracing
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/completions", post(completion));

    // TODO: load model/vocab and save into state

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::debug!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn completion(Json(payload): Json<CompletionParameters>) -> impl IntoResponse {
    let (model, vocab) = llama_rs::Model::load("models/7B/ggml-model-q4_0.bin", 512, |_| {}).unwrap();
    let mut rng = rand::thread_rng();
    // TODO: move model out of state to perform inference
    model.inference_with_prompt(
        &vocab, 
        &llama_rs::InferenceParameters {
            n_threads: 4,
            n_predict: 128,
            n_batch: 8,
            top_k: 40,
            repeat_last_n: 64,
            top_p: 0.95,
            repeat_penalty: 1.30,
            temp: 0.80,
        }, &payload.prompt, &mut rng, {
            move |t| {
                // TODO: figure out how to move out of callback to http response
                println!("{:?}", t)
            }
        });
    let response = CompletionResponse {
        prompt: payload.prompt,
        text: "response".into(), // TODO replace with model response
    };
    return (StatusCode::CREATED, Json(response))

}

#[derive(Deserialize)]
struct CompletionParameters {
    prompt: String,
}

#[derive(Serialize)]
struct CompletionResponse {
    prompt: String,
    text: String,
}
