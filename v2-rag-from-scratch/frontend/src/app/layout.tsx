import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
    title: 'StudyBuddy',
    description: 'Your AI-powered study assistant',
    icons: {
        icon: [
            { url: '/images/favicon.ico', sizes: 'any' },
            { url: '/images/favicon-96x96.png', sizes: '96x96', type: 'image/png' },
        ],
        apple: '/images/apple-touch-icon.png',
    },
    manifest: '/images/site.webmanifest',
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en">
            <body className="bg-white text-gray-900 antialiased">
                {children}
            </body>
        </html>
    )
}
