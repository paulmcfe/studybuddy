import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
    experimental: {
        serverActions: {
            bodySizeLimit: '50mb',
        },
    },
    async rewrites() {
        return [
            {
                source: '/api/:path*',
                destination: 'http://127.0.0.1:8000/api/:path*',
            },
            {
                source: '/ws/:path*',
                destination: 'http://127.0.0.1:8000/ws/:path*',
            },
        ]
    },
}

export default nextConfig
