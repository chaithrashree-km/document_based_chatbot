type Props = { role: string; text: string; source?: string; isTyping?: boolean }

export default function Message({ role, text, source, isTyping }: Props) {
  const isUser = role === "user"
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`rounded-xl px-4 py-2 max-w-[75%] ${isUser ? "bg-gray-200 text-black" : "bg-gray-100 text-gray-900"}`}>

           {isTyping ? (
          <div className="flex items-center gap-1 py-1 px-1">
            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
          </div>
        ) : (
          <>
            <p>{text}</p>
            {source && <p className="text-sm text-gray-400 mt-1">{source}</p>}
          </>
        )}

      </div>
    </div>
  )
}