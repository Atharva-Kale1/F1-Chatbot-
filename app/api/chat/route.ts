import { DataAPIClient } from '@datastax/astra-db-ts';
import { HfInference } from '@huggingface/inference';

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  HF_TOKEN,
  GROQ_API_KEY,
} = process.env;

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT, { keyspace: ASTRA_DB_NAMESPACE });
const hf = new HfInference(HF_TOKEN);

async function getHuggingFaceEmbedding(text: string): Promise<number[]> {
  const response = await hf.featureExtraction({
    model: 'BAAI/bge-small-en-v1.5',
    inputs: text
  });

  // Handle all possible response types
  if (Array.isArray(response)) {
    // Case 1: Single embedding (number[])
    if (response.length > 0 && typeof response[0] === 'number') {
      return response as number[];
    }
    // Case 2: Batch of embeddings (number[][])
    if (response.length > 0 && Array.isArray(response[0])) {
      return response[0] as number[]; // Return first embedding
    }
  }
  // Case 3: Object response (convert to array)
  if (typeof response === 'object' && response !== null) {
    return Object.values(response) as number[];
  }

  // Fallback: Return empty array if unexpected format
  console.warn('Unexpected embedding response format:', response);
  return [];
}

// Helper: stream a final string result in small chunks, to preserve "streaming" UX when we can't use HF streaming API
function streamText(controller: ReadableStreamDefaultController, text: string, emit: (c: ReadableStreamDefaultController, t: string) => void) {
  if (!text) return;
  // Chunk by ~40 chars to avoid huge payload lines
  const chunkSize = 40;
  for (let i = 0; i < text.length; i += chunkSize) {
    emit(controller, text.slice(i, i + chunkSize));
  }
}

// New: Groq non-streaming generation helper
async function generateWithGroq(messages: { role: 'system' | 'user' | 'assistant'; content: string }[]): Promise<string> {
  if (!GROQ_API_KEY) return '';
  try {
    const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${GROQ_API_KEY}`,
      },
      body: JSON.stringify({
        model: 'llama-3.1-8b-instant',
        messages: messages.map(m => ({ role: m.role, content: m.content })),
        temperature: 0.7,
        max_tokens: 256,
        stream: false,
      }),
    });
    if (!res.ok) {
      console.warn('Groq API error:', res.status, await res.text());
      return '';
    }
    const json = await res.json();
    return json?.choices?.[0]?.message?.content ?? '';
  } catch (e) {
    console.error('Groq request failed:', e);
    return '';
  }
}

// New: HF non-streaming fallbacks (text-generation, then text2text)
async function generateWithHF(prompt: string): Promise<string> {
  const textGenModels = ['bigscience/bloom-560m', 'gpt2', 'distilgpt2'];
  for (const model of textGenModels) {
    try {
      const result: any = await hf.textGeneration({
        model,
        inputs: prompt,
        parameters: {
          temperature: 0.7,
          max_new_tokens: 256,
          return_full_text: false,
        },
      });
      const text =
        typeof result === 'string'
          ? result
          : (result?.generated_text ?? result?.[0]?.generated_text ?? '');
      if (text) return text;
    } catch (e: any) {
      const msg = String(e?.message || e);
      console.warn(`HF textGeneration failed for ${model}:`, msg);
      continue;
    }
  }
  const t5Models = ['google/flan-t5-base', 'google/flan-t5-small'];
  for (const model of t5Models) {
    try {
      const result: any = await hf.textToText({
        model,
        inputs: prompt,
        parameters: {
          temperature: 0.7,
          max_new_tokens: 256,
        },
      });
      // textToText returns array of { generated_text }
      const text =
        typeof result === 'string'
          ? result
          : (result?.generated_text ?? result?.[0]?.generated_text ?? '');
      if (text) return text;
    } catch (e: any) {
      const msg = String(e?.message || e);
      console.warn(`HF textToText failed for ${model}:`, msg);
      continue;
    }
  }
  return '';
}

export async function POST(req: Request) {
  try {
    const { messages }: { messages: ChatMessage[] } = await req.json();
    const latestMessage = messages?.[messages.length - 1]?.content?.toString().trim();

    if (!latestMessage) {
      return new Response(JSON.stringify({ error: 'No user message provided.' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    let docContext = '';

    // Get embeddings using Hugging Face
    let embedding: number[] = [];
    try {
      embedding = await getHuggingFaceEmbedding(latestMessage);
    } catch (e) {
      console.warn('HF embedding error:', e);
      embedding = [];
    }

    try {
      // Only query vector DB if we have a valid embedding
      if (embedding?.length) {
        const collection = await db.collection(ASTRA_DB_COLLECTION);
        const cursor = collection.find(null, {
          sort: {
            $vector: embedding,
          },
          limit: 10,
        });
        const documents = await cursor.toArray();
        const docsMap = documents?.map((doc) => doc.text);
        docContext = JSON.stringify(docsMap);
      } else {
        console.log('Skipping DB vector search due to empty embedding.');
      }
    } catch {
      console.log('Error querying database');
      docContext = '';
    }

    const systemPrompt = `You are an AI assistant who knows everything about Formula One. Use the below context to augment what you know about Formula One racing. The context will provide you with the most recent page data from wikipedia, the official F1 website and others. If the context doesn't include the information you need answer based on your existing knowledge and don't mention the source of your information or what the context does or doesn't include. Format responses using markdown where applicable and don't return images.
    ---------
    START CONTEXT
    ${docContext}
    END CONTEXT
    ---------
    QUESTION: ${latestMessage}
    ---------
    `;

    // Format messages for downstream provider
    const groqMessages = [
      { role: 'system', content: systemPrompt },
      ...messages.map((msg) => ({
        role: msg.role,
        content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content),
      })),
    ];

    // Use Hugging Face Inference (free tier)
    if (!HF_TOKEN) {
      return new Response(JSON.stringify({ error: 'Missing HF_TOKEN in environment.' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Build a simple instruction-style prompt from the messages
    const prompt = groqMessages
      .map(m => `${m.role}: ${m.content}`)
      .join('\n') + '\nassistant:';

    // Remove HF streaming model selection; we will prefer Groq then HF non-streaming
    const encoder = new TextEncoder();

    // Helper to emit tokens in AI SDK v4-compatible "0:\"...\"" format
    const emit = (controller: ReadableStreamDefaultController, text: string) => {
      if (!text) return;
      const formatted = `0:"${text.replace(/"/g, '\\"').replace(/\n/g, '\\n')}"\n`;
      controller.enqueue(encoder.encode(formatted));
    };

    const readableStream = new ReadableStream({
      async start(controller) {
        try {
          // 1) Try Groq first if key is present
          let text = '';
          if (GROQ_API_KEY) {
            text = await generateWithGroq(groqMessages as any);
          }

          // 2) Fallback to HF non-streaming (works even if HF text-generation streaming is 404)
          if (!text && HF_TOKEN) {
            text = await generateWithHF(prompt);
          }

          if (!text) {
            emit(controller, "I couldn't generate a response right now. Please try again.");
          } else {
            streamText(controller, text, emit);
          }
          controller.close();
        } catch (fatalErr) {
          console.error('Generation pipeline error:', fatalErr);
          emit(controller, "I couldn't generate a response right now. Please try again.");
          controller.close();
        }
      },
    });

    return new Response(readableStream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });

  } catch (error) {
    console.log('Error:', error);
    return new Response(JSON.stringify({ error: 'Internal server error' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
