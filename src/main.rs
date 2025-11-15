use dotenv::dotenv;
use langchain_rust::{
    chain::{chain_trait::Chain, llm_chain::LLMChainBuilder},
    language_models::options::CallOptions,
    llm::{Deepseek, DeepseekModel},
    prompt::{PromptTemplate, TemplateFormat},
    prompt_args,
};
use std::env;

#[tokio::main]
async fn main() {
    dotenv().ok();
    // Get API key from environment variable
    let api_key =env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY environment variable must be set");
    let model=env::var("MODEL").expect("MODEL environment variable must be set");
    let base_url=env::var("BASE_URL").expect("BASE_URL environment variable must be set");

    // Setup the Deepseek client with desired model and parameters
    let deepseek = Deepseek::new()
        .with_api_key(api_key)
        .with_model(model)
        .with_base_url(base_url)
        ;

    // Create a prompt template
    let template = r#"
    你是个广州酒家的厨师，请根据用户的问题，提供一份详细的菜谱。
    
    用户问题: {问题}
    
    请提供靠谱的答案:
    "#;

    let prompt = PromptTemplate::new(
        template.to_owned(),
        vec!["问题".to_owned()],
        TemplateFormat::FString,
    );

    // Create an LLMChain using the builder pattern
    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(deepseek)
        .build()
        .unwrap();

    // Execute the chain with a question
    let inputs = prompt_args! {
        "问题" => "冬季推荐一款菜"
    };
    let result = chain.call(inputs).await.unwrap();

    println!(
        "问题：冬季推荐一款菜"
    );
    println!("\nDeepseek回复:");
    println!("{}", result.generation);
}
