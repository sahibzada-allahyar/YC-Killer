import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const token = request.cookies.get('auth_token');
  const isAuthPage = request.nextUrl.pathname === '/login';

  if (!token && !isAuthPage) {
    // Redirect to login if not authenticated and not already on login page
    return NextResponse.redirect(new URL('/login', request.url));
  }

  if (token && isAuthPage) {
    // Redirect to dashboard if authenticated and trying to access login page
    return NextResponse.redirect(new URL('/calls', request.url));
  }

  return NextResponse.next();
}

export const config = {
  // Protect all routes except login and api
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
};
