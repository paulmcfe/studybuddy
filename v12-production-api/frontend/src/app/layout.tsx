import type { Metadata } from 'next'
import './globals.css'
import { AuthProvider } from '../components/AuthContext'

export const metadata: Metadata = {
    title: 'StudyBuddy v12',
    description: 'Your AI-powered study assistant - learn any subject',
    icons: {
        icon: [
            { url: '/images/favicon.ico', sizes: 'any' },
            { url: '/images/favicon-96x96.png', sizes: '96x96', type: 'image/png' },
        ],
        apple: '/images/apple-touch-icon.png',
    },
    manifest: '/images/site.webmanifest',
}

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
            <body>
                <AuthProvider>
                    {children}
                </AuthProvider>
            </body>
        </html>
    )
}
