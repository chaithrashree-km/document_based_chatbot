"use client"

import { useTheme } from "next-themes"
import { useEffect, useState } from "react"

export default function ThemeToggle(){

 const {theme,setTheme} = useTheme()
 const [mounted,setMounted] = useState(false)

 useEffect(()=>{
  setMounted(true)
 },[])

 if(!mounted) return null

 return(

<div className="flex flex-col items-center justify-center h-full">
  <button
   onClick={()=>setTheme(theme==="dark"?"light":"dark")}
   className="w-md p-4 rounded-lg border border-gray-7500 dark:border-gray-25300 hover:bg-gray-100 dark:hover:bg-gray-800"
  >
   {theme==="dark" ? "☀ Light Mode" : "🌙 Dark Mode"}
  </button>
</div>
 )
}