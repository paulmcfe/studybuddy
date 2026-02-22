'use client'

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'

interface AuthState {
    token: string | null
    userId: string | null
    email: string | null
    isAuthenticated: boolean
    login: (email: string, password: string) => Promise<void>
    register: (email: string, password: string) => Promise<void>
    logout: () => void
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
    const [token, setToken] = useState<string | null>(null)
    const [userId, setUserId] = useState<string | null>(null)
    const [email, setEmail] = useState<string | null>(null)
    const [initialized, setInitialized] = useState(false)

    useEffect(() => {
        // Restore token from localStorage on mount
        const saved = localStorage.getItem('auth_token')
        if (saved) {
            try {
                const payload = JSON.parse(atob(saved.split('.')[1]))
                setToken(saved)
                setUserId(payload.user_id)
                setEmail(payload.email)
            } catch {
                // Invalid token, clear it
                localStorage.removeItem('auth_token')
            }
        }
        setInitialized(true)
    }, [])

    const login = async (email: string, password: string) => {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        })

        if (!response.ok) {
            const data = await response.json()
            throw new Error(data.detail || 'Login failed')
        }

        const data = await response.json()
        localStorage.setItem('auth_token', data.access_token)
        setToken(data.access_token)
        setUserId(data.user_id)
        setEmail(data.email)
    }

    const register = async (email: string, password: string) => {
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        })

        if (!response.ok) {
            const data = await response.json()
            throw new Error(data.detail || 'Registration failed')
        }

        const data = await response.json()
        localStorage.setItem('auth_token', data.access_token)
        setToken(data.access_token)
        setUserId(data.user_id)
        setEmail(data.email)
    }

    const logout = () => {
        setToken(null)
        setUserId(null)
        setEmail(null)
        localStorage.removeItem('auth_token')
    }

    // Don't render children until we've checked localStorage
    if (!initialized) {
        return null
    }

    return (
        <AuthContext.Provider value={{
            token,
            userId,
            email,
            isAuthenticated: !!token,
            login,
            register,
            logout,
        }}>
            {children}
        </AuthContext.Provider>
    )
}

export function useAuth() {
    const ctx = useContext(AuthContext)
    if (!ctx) throw new Error('useAuth must be used within AuthProvider')
    return ctx
}
