import { extendTheme } from '@chakra-ui/react';

// Create a custom Chakra UI theme that uses our CSS variables
const chakraTheme = extendTheme({
  styles: {
    global: {
      body: {
        bg: 'var(--color-bg)',
        color: 'var(--color-text)',
      },
      // Add global styles for select options
      'option': {
        bg: 'var(--color-surface-dark) !important',
        color: 'var(--color-text) !important',
        padding: '8px !important',
      },
      'select': {
        appearance: 'none',
      },
      // Global scrollbar styling
      '::-webkit-scrollbar': {
        width: '10px',
        height: '10px',
      },
      '::-webkit-scrollbar-track': {
        background: 'rgba(21, 28, 44, 0.3)',
        borderRadius: '4px',
      },
      '::-webkit-scrollbar-thumb': {
        background: 'rgba(59, 130, 246, 0.6)',
        borderRadius: '4px',
        '&:hover': {
          background: 'rgba(59, 130, 246, 0.8)',
        },
      },
      '::-webkit-scrollbar-corner': {
        background: 'transparent',
      },
    },
  },
  colors: {
    // Map CSS variables to Chakra color tokens
    brand: {
      primary: 'var(--color-primary)',
      'primary-light': 'var(--color-primary-light)',
      'primary-dark': 'var(--color-primary-dark)',
      secondary: 'var(--color-secondary)',
      'secondary-light': 'var(--color-secondary-light)',
      'secondary-dark': 'var(--color-secondary-dark)',
    },
    surface: {
      base: 'var(--color-surface)',
      light: 'var(--color-surface-light)',
      dark: 'var(--color-surface-dark)',
    },
    text: {
      base: 'var(--color-text)',
      secondary: 'var(--color-text-secondary)',
      muted: 'var(--color-text-muted)',
    },
    border: {
      base: 'var(--color-border)',
      light: 'var(--color-border-light)',
    },
    status: {
      success: 'var(--color-success)',
      warning: 'var(--color-warning)',
      error: 'var(--color-error)',
      info: 'var(--color-info)',
    },
  },
  components: {
    Card: {
      baseStyle: {
        container: {
          bg: 'surface.base',
          borderColor: 'border.base',
        },
        header: {
          borderColor: 'border.base',
        },
        footer: {
          borderColor: 'border.base',
        },
      },
    },
    Button: {
      variants: {
        primary: {
          bg: 'brand.primary',
          color: 'white',
          _hover: {
            bg: 'brand.primary-dark',
          },
        },
        secondary: {
          bg: 'surface.dark',
          color: 'text.base',
          _hover: {
            bg: 'surface.light',
          },
        },
      },
    },
    Input: {
      baseStyle: {
        field: {
          borderColor: 'border.base',
          _focus: {
            borderColor: 'brand.primary',
            boxShadow: '0 0 0 1px var(--color-primary)',
          },
        },
      },
    },
    Select: {
      baseStyle: {
        field: {
          borderColor: 'border.base',
          _focus: {
            borderColor: 'brand.primary',
            boxShadow: '0 0 0 1px var(--color-primary)',
          },
        },
        icon: {
          color: 'brand.primary',
        },
      },
      variants: {
        filled: {
          field: {
            bg: 'surface.dark',
            _hover: {
              bg: 'surface.base',
            },
            _focus: {
              bg: 'surface.base',
            },
          },
        },
        outline: {
          field: {
            bg: 'surface.dark',
            borderColor: 'border.base',
            _hover: {
              borderColor: 'brand.primary',
            },
          },
        },
      },
    },
    Modal: {
      baseStyle: {
        dialog: {
          bg: 'surface.base',
          borderColor: 'border.base',
        },
        header: {
          borderColor: 'border.base',
        },
        footer: {
          borderColor: 'border.base',
        },
      },
    },
  },
});

export default chakraTheme;
