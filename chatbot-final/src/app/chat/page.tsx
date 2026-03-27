"use client"

import Sidebar from "@/components/Sidebar"
import ChatInput from "@/components/ChatInput"
import Message from "@/components/Message"
import { useState,useRef,useEffect } from "react"
import { useRouter } from "next/navigation"

export type ChatMessage = { role: string; text: string, source: string  } 

export default function Chat(){
const[messages,setMessages]= useState<ChatMessage[]>([])
const [intents,setIntents]=useState<string[]>([])
const [isLoading, setIsLoading] = useState(false)

const bottomRef=useRef<HTMLDivElement>(null)

const router = useRouter()  

useEffect(() => {
    const token = localStorage.getItem("token")
    if (!token) {
        router.push("/login")
    }
}, [])

useEffect(()=>{
bottomRef.current?.scrollIntoView({behavior:"smooth"})
},[messages, isLoading])

async function createNewChat(){

const token = localStorage.getItem("token")

await fetch("http://localhost:8000/new_chat",{
method:"POST",
headers:{
"Authorization":`Bearer ${token}`
}
})

setMessages([])

}

return(

<div className="flex h-screen">

 <Sidebar intents={intents} createNewChat={createNewChat}/>

<div className="flex flex-col flex-1">

<div className="flex-1 overflow-y-auto p-10 space-y-6">

{messages.map((msg, i) => (
  <Message key={i} text={msg.text} role={msg.role} source={msg.source} />
))}

{isLoading && <Message role="bot" text="" isTyping={true} />}

<div ref={bottomRef}/>

</div>

<ChatInput setMessages={setMessages} setIntents={setIntents} setIsLoading={setIsLoading}/>

</div>

</div>
)
}