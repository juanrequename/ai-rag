// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

const BACKEND_URL = process.env.RAG_BACKEND_URL || 'http://localhost:8000';

export async function POST(req: Request) {
	try {
		const { question, k = 4, document_id }: { question: string; k?: number; document_id?: string } = await req.json();

		if (!question) {
			return new Response('No question provided', { status: 400 });
		}

		// Call the Python backend streaming endpoint
		const backendResponse = await fetch(`${BACKEND_URL}/rag/query/stream`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				question,
				k,
				document_id,
			}),
		});

		if (!backendResponse.ok) {
			const errorText = await backendResponse.text();
			return new Response(`Backend error: ${errorText}`, { status: backendResponse.status });
		}

		// Just proxy the backend stream directly
		return new Response(backendResponse.body, {
			headers: {
				'Content-Type': 'application/x-ndjson',
				'Cache-Control': 'no-cache',
				Connection: 'keep-alive',
			},
		});
	} catch (error) {
		console.error('chat-rag POST error', error);
		return new Response('Internal server error', { status: 500 });
	}
}
