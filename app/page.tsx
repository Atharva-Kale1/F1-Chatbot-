// import { Groq } from 'groq-sdk';
// import { streamText } from 'ai';
// import { DataAPIClient } from '@datastax/astra-db-ts';
// import { HfInference } from '@huggingface/inference';
//
// const {
//   ASTRA_DB_NAMESPACE = 'default_keyspace',
//   ASTRA_DB_COLLECTION,
//   ASTRA_DB_API_ENDPOINT,
//   ASTRA_DB_APPLICATION_TOKEN,
//   GROQ_API_KEY,
//   HF_TOKEN
// } = process.env;
//
// const groq = new Groq({ apiKey: GROQ_API_KEY });
// const hf = new HfInference(HF_TOKEN);
// const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
//
// export async function POST(req: Request) {
//   try {
//     const { messages } = await req.json();
//     const latestMessage = messages[messages.length - 1]?.content;
//
//     let docContext = '';
//
//     // Get embeddings (384 dimensions)
//     const embeddingResponse = await hf.featureExtraction({
//       model: 'BAAI/bge-small-en-v1.5',
//       inputs: latestMessage
//     });
//
//     // Handle both array and object responses
//     const embedding = Array.isArray(embeddingResponse)
//         ? embeddingResponse
//         : Object.values(embeddingResponse);
//
//     try {
//       const db = client.db(ASTRA_DB_API_ENDPOINT, {
//         keyspace: ASTRA_DB_NAMESPACE
//       });
//
//       const collection = await db.collection(ASTRA_DB_COLLECTION);
//       const documents = await collection.find(null, {
//         sort: { $vector: embedding },
//         limit: 10
//       }).toArray();
//
//       docContext = documents.map(doc => doc.text).join('\n---\n');
//     } catch (e) {
//       console.error('Vector search error:', e);
//     }
//
//     // Create the system prompt with context
//     const systemPrompt = {
//       role: 'system',
//       content: `You are an AI Formula One expert. ${
//           docContext ? `Context:\n${docContext}\n` : ''
//       }Answer in markdown without mentioning the context.`
//     };
//
//     // Use Groq directly instead of AI SDK for compatibility
//     const groqResponse = await groq.chat.completions.create({
//       messages: [systemPrompt, ...messages],
//       model: 'llama3-70b-8192',
//       temperature: 0.7,
//       max_tokens: 1024,
//       stream: true
//     });
//
//     // Convert Groq stream to ReadableStream
//     const stream = new ReadableStream({
//       async start(controller) {
//         const encoder = new TextEncoder();
//         for await (const chunk of groqResponse) {
//           const content = chunk.choices[0]?.delta?.content || '';
//           controller.enqueue(encoder.encode(content));
//         }
//         controller.close();
//       }
//     });
//
//     return new Response(stream, {
//       headers: { 'Content-Type': 'text/plain' }
//     });
//   } catch (error) {
//     console.error('API error:', error);
//     return new Response(
//         JSON.stringify({
//           error: 'Processing failed',
//           details: error instanceof Error ? error.message : String(error)
//         }),
//         { status: 500 }
//     );
//   }
// }

'use client';

import Image from 'next/image';
import logo from './assets/logo.png';
import { useChat } from 'ai/react';
import { Message } from 'ai';
import { PromptSuggestionRow } from './components/PromptSuggestionRow';
import { LoadingBubble } from './components/LoadingBubble';
import { Bubble } from './components/Bubble';

const Home = () => {
  const {
    append,
    isLoading,
    messages,
    input,
    handleInputChange,
    handleSubmit,
  } = useChat({
    api: '/api/chat'
  });

  const handlePromptClick = (prompt: string) => {
    const msg: Message = {
      id: crypto.randomUUID(),
      content: prompt,
      role: 'user',
    };
    append(msg);
  };

  const noMessages = !messages || messages.length === 0;

  return (
      <main>
        <Image src={logo} width="250" alt="logo" />
        <section className={noMessages ? '' : 'populated'}>
          {noMessages ? (
              <>
                <p className="starter-text">
                  Ask an F1 question and get the latest answers.
                </p>
                <br />
                <PromptSuggestionRow onPromptClick={handlePromptClick} />
              </>
          ) : (
              <>
                {messages.map((message, index) => (
                    <Bubble key={`message-${index}`} message={message} />
                ))}
                {isLoading && <LoadingBubble />}
              </>
          )}
        </section>
        <form onSubmit={handleSubmit}>
          <input
              className="question-box"
              onChange={handleInputChange}
              value={input}
              placeholder="Ask a question"
          />
          <input type="submit" />
        </form>
      </main>
  );
};

export default Home;