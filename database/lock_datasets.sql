ALTER TABLE public.vga_crowd_records
ADD COLUMN IF NOT EXISTS is_locked boolean NOT NULL DEFAULT false;