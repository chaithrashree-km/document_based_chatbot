"use client"

import { useState } from "react"
import ThemeToggle from "./ThemeToggle"
import Logout from "./Logout"

type SidebarProps = {
  intents: string[]
  createNewChat: () => void
}

export default function Sidebar({ intents, createNewChat }: SidebarProps){

const[open,setOpen]=useState(true)

return(

<div className="w-[320px] border-r border-gray-200 dark:border-gray-800 p-4 flex flex-col">

<button onClick={createNewChat}
className="w-full bg-gray-100 dark:bg-gray-800 text-black dark:text-white p-2 rounded-md"
>
New Chat
</button>

<div className="flex-1 overflow-y-auto mt-2 space-y-1 text-sm">
        {intents.map((intent, index) => (
          <div
            key={index}
            className="px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded cursor-pointer truncate"
          >
            {intent}
          </div>
        ))}
      </div>

<div className="mt-auto pt-12 w-full px-2">
  <div className="w-[200px] h-[60px] overflow-hidden  h-24">
    <ThemeToggle />
  </div>
  <div className="w-[200px] h-[60px] overflow-hidden  h-24">
    <Logout />
  </div>
</div>

 </div>
)
}