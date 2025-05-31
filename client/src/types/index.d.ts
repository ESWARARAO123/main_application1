// General type definitions to help TypeScript understand dynamic indexing and other patterns

// For dynamic object indexing
interface StringIndexable {
  [key: string]: any;
}

// Record is already a built-in TypeScript utility type, no need to redeclare

// Add the specific types from the error
interface UserFormData extends StringIndexable {
  username?: string;
  password?: string;
  name?: string;
  email?: string;
  role?: string;
}

// Allows explicit declaration of "as any" index
interface DynamicObject {
  [key: string]: any;
}

// Declare modules for any libraries without type definitions
declare module 'remark-gfm'; 