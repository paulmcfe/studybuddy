import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
    title: 'StudyBuddy v8',
    description: 'Your AI-powered study assistant with evaluation infrastructure',
    icons: {
        icon: [
            { url: '/images/favicon.ico', sizes: 'any' },
            { url: '/images/favicon-96x96.png', sizes: '96x96', type: 'image/png' },
        ],
        apple: '/images/apple-touch-icon.png',
    },
    manifest: '/images/site.webmanifest',
}

// Script to initialize theme before render (prevents flash)
const themeInitScript = `
(function() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = savedTheme || (prefersDark ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);
})();
`

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en" suppressHydrationWarning>
            <head>
                <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
            </head>
            <body className="bg-white text-gray-900 antialiased dark:bg-[hsl(220,15%,13%)] dark:text-[hsl(0,0%,92%)]">
                {children}
            </body>
        </html>
    )
}
