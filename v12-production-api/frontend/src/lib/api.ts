/**
 * Authenticated API fetch utility for StudyBuddy v12.
 *
 * Wraps the native fetch() to automatically include the JWT
 * Authorization header from localStorage on every request.
 */

export async function apiFetch(
    path: string,
    options: RequestInit = {}
): Promise<Response> {
    const token = localStorage.getItem('auth_token')
    const headers: Record<string, string> = {
        ...(options.headers as Record<string, string> || {}),
    }

    if (token) {
        headers['Authorization'] = `Bearer ${token}`
    }

    // Don't set Content-Type for FormData (browser sets it with boundary)
    if (!(options.body instanceof FormData)) {
        headers['Content-Type'] = headers['Content-Type'] || 'application/json'
    }

    return fetch(path, { ...options, headers })
}
