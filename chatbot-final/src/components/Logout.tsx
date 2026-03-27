"use client"

import { LogOut } from 'lucide-react'

export default function Logout(){

async function logout(){
   const token = localStorage.getItem("token")
   try{
       await fetch("http://localhost:8000/logout",{
       method:"POST",
       headers:{
            "Authorization":`Bearer ${token}`
        }
       })

       localStorage.removeItem("token")
       window.location.href="/login"

    }catch(error){
        console.error("Logout failed",error)
    }
 }
return (

<div className="flex flex-col items-center justify-center h-full">
  <button
    onClick={logout}
    className="flex items-center gap-2 w-md p-4 rounded-lg border border-gray-7500 dark:border-gray-25300 hover:bg-gray-100 dark:hover:bg-gray-800">
    <span>Logout</span>
    <LogOut size={20} />
  </button>
</div> 
)
}