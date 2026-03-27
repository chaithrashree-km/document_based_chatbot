"use client"

import { useRouter } from "next/navigation"
import { useState } from "react"

export default function Signup(){

const router=useRouter()

const[username,setUsername]=useState("")
const[email,setEmail]=useState("")
const[password,setPassword]=useState("")
const[error, setError] = useState<string | null>(null)
const[isLoading, setIsLoading] = useState(false)

function parseError(detail: any): string {
  if (!detail) return "Login failed. Please check your credentials."
  if (typeof detail === "string") return detail
  if (Array.isArray(detail)) {
    return detail.map((e: any) => e.msg ?? JSON.stringify(e)).join(", ")
  }
  return "Login failed. Please check your credentials."
}

async function signup(){
    setError(null) 

    if (!username.trim()||!email.trim() || !password.trim()) {
      setError("Please enter all the details")
      return
    }

    setIsLoading(true)

try{

const res = await fetch("http://localhost:8000/signup",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({username,email,password})
})

const data = await res.json()

if(!res.ok){
setError(parseError(data.detail))
return
}

localStorage.setItem("token",data.access_token)
router.push("/chat")
}catch(error){
console.error(error)
setError("Unable to connect. Please check your network and try again.")
}finally {
      setIsLoading(false)  
    }
}

function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter") signup()
  }

return(

<div className="flex items-center justify-center h-screen bg-gray-100 dark:bg-black">

<div className="w-[360px] bg-white dark:bg-[#1a1a1a] p-8 rounded-xl shadow-lg">

<h1 className="text-xl font-semibold text-center mb-6">
Create Account
</h1>

<input
placeholder="Username"
className="w-full p-3 mb-4 border rounded-md dark:bg-[#2a2a2a]"
onChange={(e)=>setUsername(e.target.value)}
onKeyDown={handleKeyDown}
disabled={isLoading}
/>

<input
placeholder="Email"
className="w-full p-3 mb-4 border rounded-md dark:bg-[#2a2a2a]"
onChange={(e)=>setEmail(e.target.value)}
onKeyDown={handleKeyDown}
disabled={isLoading}
/>

<input
type="password"
placeholder="Password"
className="w-full p-3 mb-6 border rounded-md dark:bg-[#2a2a2a]"
onChange={(e)=>setPassword(e.target.value)}
onKeyDown={handleKeyDown}
disabled={isLoading}
/>

<button
onClick={signup}
disabled={isLoading}
className="w-full bg-black text-white py-3 rounded-md hover:bg-gray-800 disabled:opacity-60 disabled:cursor-not-allowed"
>
{isLoading ? "Signing in..." : "Sign Up"}
</button>


<p className="h-5 mt-2 text-sm text-red-500 text-center">
{error}
</p>
</div>
</div>
)
}