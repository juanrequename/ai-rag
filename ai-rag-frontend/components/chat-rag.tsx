'use client';

import { Conversation, ConversationContent, ConversationScrollButton } from '@/components/ai-elements/conversation';
import { Loader } from '@/components/ai-elements/loader';
import { Message, MessageContent } from '@/components/ai-elements/message';
import {
	PromptInput,
	PromptInputSubmit,
	PromptInputTextarea,
	PromptInputToolbar,
	PromptInputTools,
} from '@/components/ai-elements/prompt-input';
import { Response } from '@/components/ai-elements/response';
import { Source, Sources, SourcesContent, SourcesTrigger } from '@/components/ai-elements/source';
import { useState, useCallback } from 'react';

interface ChatMessage {
	id: string;
	role: 'user' | 'assistant';
	content: string;
	sources?: Array<{ document_id: string; filename: string; chunk_index: string | number }>;
}

export default function AIChatRAG() {
	const [input, setInput] = useState('');
	const [messages, setMessages] = useState<ChatMessage[]>([]);
	const [isLoading, setIsLoading] = useState(false);

	const handleSubmit = useCallback(
		async (e: React.FormEvent) => {
			e.preventDefault();
			if (!input.trim() || isLoading) return;

			const userMessage: ChatMessage = {
				id: crypto.randomUUID(),
				role: 'user',
				content: input.trim(),
			};

			const assistantMessage: ChatMessage = {
				id: crypto.randomUUID(),
				role: 'assistant',
				content: '',
				sources: [],
			};

			setMessages((prev) => [...prev, userMessage, assistantMessage]);
			setInput('');
			setIsLoading(true);

			try {
				const response = await fetch('/api/chat-rag', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ question: userMessage.content, k: 4 }),
				});

				if (!response.ok) {
					throw new Error(await response.text());
				}

				const reader = response.body?.getReader();
				if (!reader) throw new Error('No response body');

				const decoder = new TextDecoder();
				let buffer = '';

				while (true) {
					const { done, value } = await reader.read();
					if (done) break;

					buffer += decoder.decode(value, { stream: true });
					const lines = buffer.split('\n');
					buffer = lines.pop() || '';

					for (const line of lines) {
						if (!line.trim()) continue;

						try {
							const data = JSON.parse(line);

							if (data.type === 'token') {
								setMessages((prev) =>
									prev.map((msg) =>
										msg.id === assistantMessage.id ? { ...msg, content: msg.content + data.content } : msg,
									),
								);
							} else if (data.type === 'sources') {
								setMessages((prev) =>
									prev.map((msg) => (msg.id === assistantMessage.id ? { ...msg, sources: data.sources } : msg)),
								);
							} else if (data.type === 'error') {
								setMessages((prev) =>
									prev.map((msg) =>
										msg.id === assistantMessage.id ? { ...msg, content: `Error: ${data.message}` } : msg,
									),
								);
							}
						} catch {
							// Skip invalid JSON
						}
					}
				}
			} catch (error) {
				setMessages((prev) =>
					prev.map((msg) =>
						msg.id === assistantMessage.id
							? { ...msg, content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}` }
							: msg,
					),
				);
			} finally {
				setIsLoading(false);
			}
		},
		[input, isLoading],
	);

	return (
		<div className="relative mx-auto size-full h-screen max-h-[700px] max-w-4xl rounded-lg border p-6">
			<div className="flex h-full flex-col">
				<Conversation className="h-full">
					<ConversationContent>
						{messages.map((message) => (
							<Message from={message.role} key={message.id}>
								<MessageContent>
									<Response>{message.content}</Response>
									{message.sources && message.sources.length > 0 && (
										<Sources>
											<SourcesTrigger count={message.sources.length} />
											<SourcesContent>
												{message.sources.map((source, i) => (
													<Source
														key={`${message.id}-source-${i}`}
														title={`${source.filename} [document_id: ${source.document_id}, chunk: ${source.chunk_index}]`}
													/>
												))}
											</SourcesContent>
										</Sources>
									)}
								</MessageContent>
							</Message>
						))}
						{isLoading && <Loader />}
					</ConversationContent>
					<ConversationScrollButton />
				</Conversation>

				<PromptInput onSubmit={handleSubmit} className="mt-4">
					<PromptInputTextarea
						onChange={(e) => setInput(e.target.value)}
						value={input}
						placeholder="Ask a question about your documents..."
					/>
					<PromptInputToolbar>
						<PromptInputTools />
						<PromptInputSubmit disabled={!input.trim() || isLoading} />
					</PromptInputToolbar>
				</PromptInput>
			</div>
		</div>
	);
}
