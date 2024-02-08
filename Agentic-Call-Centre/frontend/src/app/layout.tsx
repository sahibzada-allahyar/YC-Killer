'use client';

import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { usePathname } from 'next/navigation';
import { AuthProvider, useAuth } from '@/contexts/AuthContext';
import Navigation from '@/components/Navigation';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'AI Call Center Dashboard',
  description: 'Admin dashboard for AI-driven call center system',
};

function AppContent({ children }: { children: React.ReactNode }) {
  const { isLoading } = useAuth();
  const pathname = usePathname();
  const isLoginPage = pathname === '/login';

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  return (
    <>
      {!isLoginPage && <Navigation />}
      <main className={`min-h-screen bg-gray-50 ${!isLoginPage ? 'pt-16' : ''}`}>
        {children}
      </main>
    </>
  );
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AuthProvider>
          <AppContent>{children}</AppContent>
        </AuthProvider>
      </body>
    </html>
  );
}
