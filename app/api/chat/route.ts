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

export async function POST(req: Request) {
  try {
    const { messages }: { messages: ChatMessage[] } = await req.json();
    const latestMessage = messages[messages.length - 1]?.content;

    let docContext = '';

    // Get embeddings using Hugging Face
    const embedding = await getHuggingFaceEmbedding(latestMessage);

    try {
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

    // Format messages for Groq API
    const groqMessages = [
      { role: 'system', content: systemPrompt },
      ...messages.map((msg) => ({
        role: msg.role,
        content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content),
      })),
    ];

    // Call Groq API directly
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${GROQ_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'llama3-70b-8192',
        messages: groqMessages,
        temperature: 0.7,
        max_tokens: 2000,
        stream: true,
      }),
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error('Groq API error:', response.status, errorData);
      throw new Error(`Groq API error: ${response.status} ${errorData}`);
    }

    // Create a streaming response compatible with AI SDK v4
    const readableStream = new ReadableStream({
      async start(controller) {
        const reader = response.body?.getReader();
        if (!reader) {
          controller.close();
          return;
        }

        const decoder = new TextDecoder();
        const encoder = new TextEncoder();

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6).trim();
                if (data === '[DONE]') {
                  controller.close();
                  return;
                }

                try {
                  const parsed = JSON.parse(data);
                  const content = parsed.choices?.[0]?.delta?.content;
                  if (content) {
                    console.log('Sending content:', content);
                    const formattedChunk = `0:"${content.replace(/"/g, '\\"').replace(/\n/g, '\\n')}"\n`;
                    controller.enqueue(encoder.encode(formattedChunk));
                  }
                } catch {
                  // Skip invalid JSON lines
                  continue;
                }
              }
            }
          }
        } catch (error) {
          console.error('Stream processing error:', error);
          controller.error(error);
        } finally {
          reader.releaseLock();
        }
      },
    });

    return new Response(readableStream, {
      headers: {
        'Content-Type': 'text/x-unknown; charset=utf-8',
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