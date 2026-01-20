import ChatRAG from '@/components/chat-rag';
import { ModeSwitcher } from '@/components/mode-switcher';

export default function RAGPage() {
	return (
		<>
			<header className="flex justify-end p-4">
				<ModeSwitcher />
			</header>
			<ChatRAG />
		</>
	);
}
