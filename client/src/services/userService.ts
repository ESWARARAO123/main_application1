import { api } from './api';

export interface User {
  id: number;
  username: string;
  name: string;
  email: string;
  role: string;
}

const userService = {
  getUsers: async (): Promise<User[]> => {
    try {
      const response = await api.get('/users');
      return response.data;
    } catch (error) {
      console.error('Error fetching users:', error);
      return [];
    }
  },
  
  getUser: async (id: number): Promise<User | null> => {
    try {
      const response = await api.get(`/users/${id}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching user ${id}:`, error);
      return null;
    }
  },
  
  createUser: async (userData: Omit<User, 'id'>): Promise<User> => {
    const response = await api.post('/users', userData);
    return response.data;
  },
  
  updateUser: async (id: number, userData: Partial<User>): Promise<User> => {
    const response = await api.put(`/users/${id}`, userData);
    return response.data;
  },
  
  deleteUser: async (id: number): Promise<void> => {
    await api.delete(`/users/${id}`);
  }
};

export default userService; 