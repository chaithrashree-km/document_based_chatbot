"use client"
import { createContext,useContext,useState } from "react"

type Message={role:"user"|"assistant";content:string}

type ChatContextType={
 messages:Message[]
 addMessage:(msg:Message)=>void
}

const ChatContext=createContext<ChatContextType|null>(null)

export function ChatProvider({children}:{children:React.ReactNode}){

 const [messages,setMessages]=useState<Message[]>([])

 const addMessage=(msg:Message)=>{
  setMessages(prev=>[...prev,msg])
 }

 return(
  <ChatContext.Provider value={{messages,addMessage}}>
   {children}
  </ChatContext.Provider>
 )
}

export const useChat=()=>{
 const ctx=useContext(ChatContext)
 if(!ctx) throw new Error("ChatContext missing")
 return ctx
}