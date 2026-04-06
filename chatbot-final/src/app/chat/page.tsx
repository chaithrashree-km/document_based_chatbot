"use client"

import Sidebar from "@/components/Sidebar"
import ChatInput from "@/components/ChatInput"
import Message from "@/components/Message"
import { useState, useRef, useEffect, useCallback } from "react"
import { useRouter } from "next/navigation"

export type ChatMessage = { role: string; text: string; source: string }

export type SidebarEntry = {
  intent: string
  question: string
  session_id: string
}

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [entries, setEntries] = useState<SidebarEntry[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const bottomRef = useRef<HTMLDivElement>(null)
  const sessionCreated = useRef(false)
  const router = useRouter()

  useEffect(() => {
    const token = localStorage.getItem("token")
    if (!token) router.push("/login")
  }, [])

  useEffect(() => {
    if (sessionCreated.current) return
    sessionCreated.current = true
    const token = localStorage.getItem("token")
    if (!token) return
    fetch("http://localhost:8000/new_chat", {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => r.json())
      .then(data => { if (data.session_id) setActiveSessionId(data.session_id) })
      .catch(console.error)
  }, [])

  useEffect(() => {
    const token = localStorage.getItem("token")
    if (!token) return

    fetch("http://localhost:8000/get_sessions_by_user", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(res => res.json())
      .then((sessions: SidebarEntry[]) => {
        if (Array.isArray(sessions)) {
          setEntries(sessions.reverse()) // newest first
        }
      })
      .catch(err => console.error("Failed to load sessions:", err))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isLoading])

  const addEntry = useCallback((intent: string, question: string, session_id: string) => {
    setEntries(prev => [{ intent, question, session_id }, ...prev])
  }, [])

  async function loadSession(entry: SidebarEntry) {
    const token = localStorage.getItem("token")
    if (!token) return

    setActiveSessionId(entry.session_id)
    setIsLoading(true)

    try {
      const res = await fetch(
        `http://localhost:8000/get_chats_by_session?session_id=${entry.session_id}`,
        { headers: { Authorization: `Bearer ${token}` } }
      )
      if (!res.ok) throw new Error("Failed to fetch session")
      const rows: [string, string][] = await res.json()
      const loaded: ChatMessage[] = rows.flatMap(([q, r]) => [
        { role: "user", text: q, source: "" },
        { role: "bot",  text: r, source: "" },
      ])
      setMessages(loaded)
    } catch (err) {
      console.error("Could not load session:", err)
    } finally {
      setIsLoading(false)
    }
  }

  async function deleteEntry(entry: SidebarEntry) {
    const token = localStorage.getItem("token")
    if (!token) return

    try {
      const res = await fetch(
        `http://localhost:8000/delete_message?session_id=${encodeURIComponent(entry.session_id)}&question=${encodeURIComponent(entry.question)}`,
        { method: "DELETE", headers: { Authorization: `Bearer ${token}` } }
      )
      if (!res.ok) {
        console.error("delete_message failed:", await res.json())
        return
      }
    } catch (err) {
      console.error("Network error deleting message:", err)
      return
    }

    setEntries(prev => {
      let removed = false
      return prev.filter(e => {
        if (!removed && e.session_id === entry.session_id && e.question === entry.question) {
          removed = true
          return false
        }
        return true
      })
    })

    if (activeSessionId === entry.session_id) {
      setMessages(prev =>
        prev.filter((m, i, arr) =>
          !(m.role === "user" && m.text === entry.question) &&
          !(i > 0 && arr[i - 1].role === "user" && arr[i - 1].text === entry.question)
        )
      )
    }
  }

  async function createNewChat() {
    const token = localStorage.getItem("token")
    if (!token) return
    try {
      const res = await fetch("http://localhost:8000/new_chat", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      })
      const data = await res.json()
      if (data.session_id) {
        setActiveSessionId(data.session_id)
        setMessages([])
      }
    } catch (err) {
      console.error("Could not create new chat:", err)
    }
  }

  return (
    <div className="flex h-screen">
      <Sidebar
        entries={entries}
        activeSessionId={activeSessionId}
        createNewChat={createNewChat}
        onSelectEntry={loadSession}
        onDeleteEntry={deleteEntry}
      />
      <div className="flex flex-col flex-1">
        <div className="flex-1 overflow-y-auto p-10 space-y-6">
          {messages.length === 0 && !isLoading && (
            <p className="text-center text-gray-400 mt-20 text-sm">
              Start a conversation or select one from the sidebar.
            </p>
          )}
          {messages.map((msg, i) => (
            <Message key={i} text={msg.text} role={msg.role} source={msg.source} />
          ))}
          {isLoading && <Message role="bot" text="" isTyping={true} />}
          <div ref={bottomRef} />
        </div>
        <ChatInput
          setMessages={setMessages}
          onNewEntry={addEntry}
          setIsLoading={setIsLoading}
          activeSessionId={activeSessionId}
        />
      </div>
    </div>
  )
}
