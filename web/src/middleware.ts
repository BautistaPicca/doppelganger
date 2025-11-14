import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export default function middleware(req: NextRequest) {
  const authCookie = req.cookies.get("auth")?.value;

  // Rutas protegidas
  const protectedRoutes = ["/dashboard"];

  const isProtected = protectedRoutes.some((route) =>
    req.nextUrl.pathname.startsWith(route)
  );

  if (isProtected && !authCookie) {
    const loginUrl = new URL("/auth/login", req.url);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

// A qué rutas se aplica el middleware
export const config = {
  matcher: ["/dashboard/:path*"],
};
