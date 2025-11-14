const API_URL = 'http://localhost:5000';

export const setToken = (token) => {
  if (typeof window !== 'undefined') {
    localStorage.setItem('token', token);
    document.cookie = "auth=1; path=/; max-age=3600; SameSite=Lax;";
  }
  
};

export const getToken = () => {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('token');
  }
  return null;
};

export const removeToken = () => {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('token');
    document.cookie = "auth=; path=/; max-age=0";
  }
};

export const setUser = (user) => {
  if (typeof window !== 'undefined') {
    localStorage.setItem('user', JSON.stringify(user));
  }
};

export const getUser = () => {
  if (typeof window !== 'undefined') {
    const user = localStorage.getItem('user');
    return user ? JSON.parse(user) : null;
  }
  return null;
};

export const logout = () => {
  removeToken();
  if (typeof window !== 'undefined') {
    localStorage.removeItem('user');
  }
};

export const isAuthenticated = () => {
  return getToken() !== null;
};

export const fetchWithAuth = async (url, options = {}) => {
  const token = getToken();
  
  const headers = {
    ...options.headers,
  };
  
  if (token && !options.body instanceof FormData) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  const response = await fetch(`${API_URL}${url}`, {
    ...options,
    headers: token ? { ...headers, 'Authorization': `Bearer ${token}` } : headers,
  });
  
  if (response.status === 401) {
    logout();
    if (typeof window !== 'undefined') {
      window.location.href = '/auth/login';
    }
  }
  
  return response;
};