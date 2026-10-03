# Cloud Backup (Encrypted, Opt-In)

TrueMemory can copy your local memory database to S3-compatible object storage
as an encrypted backup, and restore it later. This is the Phase 1 slice of
issue #199: one-way backup plus restore. It is fully opt-in. If you never
configure it, TrueMemory stays 100% local and nothing ever leaves your machine.

## What it does

- Creates a consistent snapshot of your live SQLite database (WAL content is
  folded in, so the snapshot is never torn).
- Validates the snapshot, encrypts it with AES-256-GCM using a key that only
  you control, and uploads only the encrypted artifact.
- Restores on any machine: downloads the backup, decrypts it, validates that it
  is a healthy TrueMemory database, protects your current database with a
  safety snapshot, and only then replaces it.

## What it does NOT do

Phase 1 is deliberately minimal. There is:

- No bidirectional sync (backups are one-way: local to cloud)
- No conflict resolution or multi-device merging
- No LLM categorization or selective sharing
- No agent collaboration, team features, dashboards, or billing
- No automatic uploading: backup and restore only run when you ask for them

## Requirements

- An S3-compatible bucket (AWS S3, Cloudflare R2, MinIO, or any S3-compatible
  endpoint)
- The optional dependency: `pip install "truememory[backup]"` (adds boto3)
- A 32-byte encryption key that you generate and keep

## Configuration

All settings come from environment variables. Nothing is written to
`~/.truememory/config.json`, and no upload ever happens without
`TRUEMEMORY_BACKUP_ENABLED=1`.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TRUEMEMORY_BACKUP_ENABLED` | yes | `0` | Must be `1`/`true`/`yes`/`on` to enable cloud backup |
| `TRUEMEMORY_BACKUP_BUCKET` | yes | | Bucket name |
| `TRUEMEMORY_BACKUP_KEY` | yes | | Base64 32-byte encryption key (see below) |
| `TRUEMEMORY_BACKUP_ENDPOINT` | no | AWS endpoint | Custom endpoint for R2/MinIO, e.g. `https://ACCOUNTID.r2.cloudflarestorage.com` |
| `TRUEMEMORY_BACKUP_REGION` | no | `us-east-1` | Region name. S3-compatible providers usually ignore it, but boto3 requires one |
| `TRUEMEMORY_BACKUP_PREFIX` | no | `truememory/` | Object key prefix |
| `TRUEMEMORY_BACKUP_OBJECT` | no | `<prefix><db filename>.tmbackup` | Explicit object key override (use when restoring to a differently named file) |
| `TRUEMEMORY_BACKUP_ACCESS_KEY_ID` | no | provider chain | Explicit access key. When omitted, boto3's standard chain is used (`AWS_*` env vars, shared credentials file, instance role) |
| `TRUEMEMORY_BACKUP_SECRET_ACCESS_KEY` | no | provider chain | Secret for the access key above |

Example (fake values only, never commit real credentials):

```bash
export TRUEMEMORY_BACKUP_ENABLED=1
export TRUEMEMORY_BACKUP_BUCKET=my-bucket
export TRUEMEMORY_BACKUP_ENDPOINT=https://1234567890abcdef.r2.cloudflarestorage.com
export TRUEMEMORY_BACKUP_REGION=auto
export TRUEMEMORY_BACKUP_KEY="ZmFrZS1rZXktMzJieXRlLWV4YW1wbGUtMDAwMDAwMDA="
```

## The encryption key

Generate one with:

```bash
truememory-ingest backup --generate-key
```

This prints a random 32-byte key as base64. Store it in a password manager or
offline note, then set it as `TRUEMEMORY_BACKUP_KEY` (or pass `--key-file`).

Facts you must know before using this feature:

- The key is generated on your machine and never leaves it. It is not uploaded,
  not stored in bucket metadata, and not sent anywhere.
- Every backup is encrypted with AES-256-GCM before upload. The cloud provider
  only ever sees ciphertext.
- Losing the key means every existing backup is permanently undecryptable.
  There is no recovery, no key escrow, and no reset.
- The same key must be used for restore. Restoring with a wrong key fails
  cleanly and never touches your current database.
- Do not commit the key, put it in shell history, or log it.
- The default object key includes your local database filename (for example
  `truememory/memories.db.tmbackup`), which is visible to the storage provider.
  If the filename itself is sensitive, set `TRUEMEMORY_BACKUP_OBJECT` to an
  opaque key.

## Usage

Backup:

```bash
truememory-ingest backup                  # snapshots, encrypts, uploads
truememory-ingest backup --db /path/to/memories.db
truememory-ingest backup --key-file /path/to/key.txt
```

Restore (replaces the local database, asks for confirmation first):

```bash
truememory-ingest restore                 # interactive confirmation
truememory-ingest restore --yes           # non-interactive
truememory-ingest restore --db /path/to/memories.db --key-file /path/to/key.txt
```

Python API:

```python
from truememory.cloud_backup import BackupConfig, backup_database, restore_database

config = BackupConfig.from_env()          # reads TRUEMEMORY_BACKUP_* variables
result = backup_database(config)           # -> BackupResult(bucket, key, ...)
restored = restore_database(config)        # -> RestoreResult(..., safety_backup=...)
```

Both functions accept an explicit `db_path=` and `key=` for programmatic use.

## How restore protects your data

Restore is designed so that every failure mode leaves your current database
intact:

1. Download the encrypted object to a temporary directory.
2. Decrypt it to a staging file next to the target database.
3. Validate it: SQLite `quick_check` plus a check that the `messages` table
   exists. Anything invalid is rejected here, before your database is touched.
4. Snapshot your current database to a `*.backup-pre-restore-*` file using the
   same consistent-snapshot mechanism used for pre-migration backups. If this
   snapshot cannot be created, the restore aborts.
5. Remove stale `-wal`/`-shm`/`-journal` files from the old database. If they
   cannot be removed (for example a live process holds them on Windows), the
   restore aborts and your database is unchanged.
6. Atomically replace the database file (`os.replace` with retry for Windows
   sharing violations).
7. Re-open and re-validate the restored database. If this post-replacement
   validation fails (for example a transient I/O fault during the finalize),
   the pre-restore snapshot is copied back automatically and the restore
   reports failure with the snapshot path.

If the database replacement itself fails, TrueMemory automatically copies the
pre-restore snapshot back over the database and reports the snapshot path.

Limitations to be aware of:

- Restore replaces the whole database. Anything stored after the backup was
  taken is gone from the live database (it survives only in the pre-restore
  snapshot).
- Run restore while all TrueMemory processes (MCP server, hooks, other CLIs)
  are stopped. On Windows a live process holds the database file open and the
  replacement will fail; on POSIX the replacement succeeds but the still-running
  process keeps writing to the replaced-out file, losing those writes.
- Atomic replacement is atomic within one filesystem; the staging file is
  created in the same directory as the database for that reason.

## Backup format

Encrypted backups are versioned, self-describing files:

```
offset  size  field
0       5     magic "TMBAK"
5       1     format version (currently 1)
6       1     algorithm id (1 = AES-256-GCM)
7       12    random nonce
19      N     ciphertext
19+N    16     GCM authentication tag
```

The header is bound into the authentication tag as additional data, so edits
to the version, algorithm, or nonce are detected. A wrong key, a corrupted
upload, or a truncated file all fail decryption with an explicit error. Future
format versions can extend this header without breaking old restores.

## Security summary

- The snapshot is created locally with SQLite's Online Backup API. It is
  validated before encryption.
- Encryption (AES-256-GCM, random 96-bit nonce per backup) happens on your
  machine, before any network traffic. The plaintext snapshot is deleted before
  the upload starts; if that deletion fails, the upload is aborted.
- The cloud provider receives only ciphertext. Object metadata never contains
  memory contents or the key.
- Cloud credentials are read from environment variables or the provider's
  standard chain. They are never logged or embedded in error messages.
- The encryption key is excluded from `repr()` output and error messages.
- Tests use fake credentials and an in-memory fake store. The default test
  suite never contacts a real cloud provider.

## Troubleshooting

| Error | Meaning |
|-------|---------|
| `cloud backup is not enabled` | `TRUEMEMORY_BACKUP_ENABLED` is not set to a truthy value. This is the default, opt-in state |
| `cloud backup requires boto3` | Install the optional extra: `pip install "truememory[backup]"` |
| `backup could not be decrypted: wrong key or corrupted/tampered backup` | The `TRUEMEMORY_BACKUP_KEY` does not match the backup, or the object was corrupted. Your local database is untouched |
| `no backup object at s3://...` | Nothing was uploaded under that key. Check bucket, prefix, or set `TRUEMEMORY_BACKUP_OBJECT` |
| `could not remove stale SQLite sidecar file` | A live TrueMemory process holds the `-wal` file. Stop all processes and retry |
| `could not replace ... with the restored database` | The target file was locked. The pre-restore snapshot path is reported in the error |
