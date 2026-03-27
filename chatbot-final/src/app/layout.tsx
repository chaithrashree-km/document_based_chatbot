"use client"

import "./globals.css"
import { ThemeProvider } from "next-themes"
import { ChatProvider } from "../context/ChatContext"

export default function RootLayout({
 children,
}: {
 children: React.ReactNode
}) {

 return (
  <html suppressHydrationWarning>
   <body>

    <ThemeProvider attribute="class" defaultTheme="system">

      <ChatProvider>
        {children}
      </ChatProvider>

    </ThemeProvider>

   </body>
  </html>
 )
}
