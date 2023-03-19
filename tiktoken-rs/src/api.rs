use anyhow::{anyhow, bail, Result};

use crate::{
    cl100k_base,
    model::get_context_size,
    p50k_base, p50k_edit, r50k_base,
    tokenizer::{get_tokenizer, Tokenizer},
    CoreBPE,
};


/// Calculates the maximum number of tokens available for completion based on the model and prompt provided.
///
/// This function determines the number of tokens left for a completion task, given the model and a prompt string.
/// It first retrieves the context size for the given model and the `CoreBPE` instance for tokenization.
/// Then, it calculates the number of tokens in the prompt using the appropriate tokenizer.
///
/// # Arguments
///
/// * `model` - A string slice representing the model name, e.g., "gpt-3.5-turbo".
/// * `prompt` - A string slice containing the prompt text.
///
/// # Errors
///
/// This function returns an error in the following cases:
///
/// * If there is a failure in creating a `CoreBPE` instance for the specified model.
///
/// # Examples
///
/// ```
/// use tiktoken_rs::get_completion_max_tokens;
///
/// let model = "gpt-3.5-turbo";
/// let prompt = "Translate the following English text to French: '";
/// let max_tokens = get_completion_max_tokens(model, prompt).unwrap();
/// ```
///
/// # Returns
///
/// If successful, the function returns a `Result` containing the maximum number of tokens available for completion,
/// based on the given model and prompt.
pub fn get_completion_max_tokens(model: &str, prompt: &str) -> Result<usize> {
    let context_size = get_context_size(model);
    let bpe = get_bpe_from_model(model)?;
    let prompt_tokens = bpe.encode_with_special_tokens(prompt).len();
    Ok(context_size.saturating_sub(prompt_tokens))
}


/// Returns a `CoreBPE` instance corresponding to the tokenizer used by the given model.
///
/// This function first retrieves the tokenizer associated with the specified model name
/// and then maps the tokenizer to the appropriate `CoreBPE` instance, which is used for
/// tokenization in different models.
///
/// # Arguments
///
/// * `model` - A string slice representing the model name for which a `CoreBPE` instance should be retrieved.
///
/// # Errors
///
/// This function returns an error if:
/// * No tokenizer is found for the given model.
/// * There is a failure in creating a `CoreBPE` instance for the tokenizer.
///
/// # Examples
///
/// ```
/// use tiktoken_rs::get_bpe_from_model;
///
/// let model = "gpt-4-0314";
/// let bpe = get_bpe_from_model(model).unwrap();
/// ```
///
/// # Returns
///
/// If successful, the function returns a `Result` containing the `CoreBPE` instance corresponding to the tokenizer used by the given model.
pub fn get_bpe_from_model(model: &str) -> Result<CoreBPE> {
    let tokenizer =
        get_tokenizer(model).ok_or_else(|| anyhow!("No tokenizer found for model {}", model))?;
    let bpe = get_bpe_from_tokenizer(tokenizer)?;
    Ok(bpe)
}

/// Returns a `CoreBPE` instance corresponding to the given tokenizer.
///
/// This function is responsible for mapping a `Tokenizer` enum variant to the appropriate
/// `CoreBPE` instance, which is used for tokenization in different models.
///
/// # Arguments
///
/// * `tokenizer` - A `Tokenizer` enum variant representing the tokenizer for which a `CoreBPE` instance should be retrieved.
///
/// # Errors
///
/// This function returns an error if there is a failure in creating a `CoreBPE` instance for the specified tokenizer.
///
/// # Examples
///
/// ```
/// use tiktoken_rs::get_bpe_from_tokenizer;
/// use tiktoken_rs::tokenizer::Tokenizer;
///
/// let tokenizer = Tokenizer::Cl100kBase;
/// let bpe = get_bpe_from_tokenizer(tokenizer).unwrap();
/// ```
///
/// # Returns
///
/// If successful, the function returns a `Result` containing the `CoreBPE` instance corresponding to the given tokenizer.
pub fn get_bpe_from_tokenizer(tokenizer: Tokenizer) -> Result<CoreBPE> {
    match tokenizer {
        Tokenizer::Cl100kBase => cl100k_base(),
        Tokenizer::R50kBase => r50k_base(),
        Tokenizer::P50kBase => p50k_base(),
        Tokenizer::P50kEdit => p50k_edit(),
        Tokenizer::Gpt2 => r50k_base(),
    }
}
