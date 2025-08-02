import {DataAPIClient} from "@datastax/astra-db-ts"
import { PuppeteerWebBaseLoader } from "langchain/document_loaders/web/puppeteer";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import OpenAI from "openai"

import  "dotenv/config"

type SimilarityMetric = "dot_product" | "cosine" | "euclidean"

const {ASTRA_DB_NAMESPACE,ASTRA_DB_COLLECTION,ASTRA_DB_API_ENDPOINT,ASTRA_DB_APPLICATION_TOKEN,OPENAI_API_KEY}=process.env

const openAI = new OpenAI({apiKey: OPENAI_API_KEY})

const f1Data = [
    'https://en.wikipedia.org/wiki/Formula_One',
    'https://www.formula1.com/en/latest.html',
    'https://en.wikipedia.org/wiki/2023_Formula_One_World_Championship',
    'https://en.wikipedia.org/wiki/2022_Formula_One_World_Championship',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_World_Drivers%27_Champions',
    'https://en.wikipedia.org/wiki/2024_Formula_One_World_Championship',
    'https://www.formula1.com/en/results.html/2024/races.html',
    'https://www.formula1.com/en/racing/2024.html'
];

const client =new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
const db= client.db(ASTRA_DB_API_ENDPOINT,{ keyspace :ASTRA_DB_NAMESPACE});

const splitter =new RecursiveCharacterTextSplitter({
    chunkSize : 512,
    chunkOverlap:100
})

const createCollection=async (similarityMetric:SimilarityMetric="dot_product")=>{
    const res=await db.createCollection(ASTRA_DB_COLLECTION,{
        vector:{
            dimension:768,
            metric:similarityMetric
        }
    })
    console.log(res)
}
async function getOllamaEmbedding(text: string): Promise<number[]> {
    const response = await fetch('http://localhost:11434/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: 'nomic-embed-text', // or another embedding model you have pulled
            prompt: text
        })
    });
    const data = await response.json();
    return data.embedding; // adjust if Ollama's response structure differs
}

const loadSampleData= async ()=>{
    const collection=await db.collection(ASTRA_DB_COLLECTION)

    for await (const url of f1Data){
        const content =await scrapePage(url)
        const chunks=await splitter.splitText(content)
        // for await (const chunk of chunks){
        //     const embedding=await openAI.embeddings.create({
        //         model:"text-embedding-3-small",
        //         input:chunk,
        //         encoding_format:"float"
        //     })
        //
        //     const vector=embedding.data[0].embedding
        //
        //     const res=await collection.insertOne({
        //         $vector:vector,
        //         text:chunk,
        //     })
        //     console.log(res)
        // }
        for await (const chunk of chunks) {
            const vector = await getOllamaEmbedding(chunk);

            const res = await collection.insertOne({
                $vector: vector,
                text: chunk,
            });
            console.log(res);
        }
    }
}

const scrapePage = async (url:string) => {
    const loader =new PuppeteerWebBaseLoader(url,{
        launchOptions:{
            headless:true,

        },
        gotoOptions:{
            waitUntil : "domcontentloaded"
        },
        evaluate: async (page,browser)=>{
            const res=await page.evaluate(()=>document.body.innerHTML)
            await browser.close()
            return res
        }
    })
    return (await loader.scrape())?.replace(/<[^>]*>/gm,'')
}
createCollection().then((collection)=>loadSampleData())