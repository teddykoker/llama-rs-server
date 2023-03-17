use flume::Receiver;
use std::sync::{Arc, Barrier};

use llama_rs::LoadProgress;

use crate::ThreadGenerateRequest;
use crate::Token;

pub fn start_model_thread(
    receiver: Receiver<ThreadGenerateRequest>,
    model_path: String,
    num_threads: i32,
    num_ctx_tokens: i32,
) {
    let barrier = Arc::new(Barrier::new(2));
    std::thread::spawn({
        let barrier = barrier.clone();
        move || {
            let (model, vocab) =
                llama_rs::Model::load(model_path, num_ctx_tokens, print_progress).unwrap();

            barrier.wait();

            let mut rng = rand::thread_rng();
            loop {
                if let Ok(ThreadGenerateRequest(request, token_tx)) = receiver.try_recv() {
                    model.inference_with_prompt(
                        &vocab,
                        &llama_rs::InferenceParameters {
                            n_threads: num_threads,
                            n_predict: request.max_tokens,
                            n_batch: request.batch_size,
                            top_k: request.top_k.try_into().unwrap(),
                            top_p: request.top_p,
                            repeat_last_n: request.repeat_last_n,
                            repeat_penalty: request.repeat_penalty,
                            temp: request.temp,
                        },
                        &request.prompt,
                        &mut rng,
                        {
                            let token_tx = token_tx.clone();
                            move |t| {
                                token_tx
                                    .send(match t {
                                        llama_rs::OutputToken::Token(t) => {
                                            Token::Token(t.to_string())
                                        }
                                        llama_rs::OutputToken::EndOfText => Token::EndOfText,
                                    })
                                    .unwrap();
                            }
                        },
                    );
                };

                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    });
    barrier.wait();
}

pub fn print_progress(progress: LoadProgress) {
    match progress {
        LoadProgress::HyperParamsLoaded(hparams) => {
            println!("Loaded HyperParams {hparams:#?}")
        }
        LoadProgress::BadToken { index } => {
            println!("Warning: Bad token in vocab at index {index}")
        }
        LoadProgress::ContextSize { bytes } => println!(
            "ggml ctx size = {:.2} MB\n",
            bytes as f64 / (1024.0 * 1024.0)
        ),
        LoadProgress::MemorySize { bytes, n_mem } => println!(
            "Memory size: {} MB {}",
            bytes as f32 / 1024.0 / 1024.0,
            n_mem
        ),
        LoadProgress::PartLoading {
            file,
            current_part,
            total_parts,
        } => println!(
            "Loading model part {}/{} from '{}'\n",
            current_part,
            total_parts,
            file.to_string_lossy(),
        ),
        LoadProgress::PartTensorLoaded {
            current_tensor,
            tensor_count,
            ..
        } => {
            if current_tensor % 8 == 0 {
                println!("Loaded tensor {current_tensor}/{tensor_count}");
            }
        }
        LoadProgress::PartLoaded {
            file,
            byte_size,
            tensor_count,
        } => {
            println!("Loading of '{}' complete", file.to_string_lossy());
            println!(
                "Model size = {:.2} MB / num tensors = {}",
                byte_size as f64 / 1024.0 / 1024.0,
                tensor_count
            );
        }
    }
}
