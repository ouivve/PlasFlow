import { test, expect } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const sampleFasta = path.resolve(
  __dirname,
  './fixtures/smoke-small.fasta'
)

test.describe('PlasFlow v2 E2E', () => {
  test('shows validation message when file is missing', async ({ page }) => {
    await page.goto('/')

    await expect(page.getByRole('heading', { name: 'PlasFlow v2' })).toBeVisible()
    await page.getByRole('button', { name: 'Start Job' }).click()

    await expect(page.getByText('Please choose a FASTA file.')).toBeVisible()
  })

  test('runs v2 classification and exposes downloadable artifacts', async ({ page, request }) => {
    await page.goto('/')

    await expect(page.getByRole('heading', { name: 'PlasFlow v2' })).toBeVisible()

    const fileInput = page.locator('label:has-text("FASTA File") input[type="file"]')
    await fileInput.setInputFiles(sampleFasta)

    await page.locator('label:has-text("Mode") select').selectOption('v2')
    await page.locator('label:has-text("Task") select').selectOption('legacy28')

    await page.getByRole('button', { name: 'Start Job' }).click()

    const statusChip = page.locator('.status')
    await expect(statusChip).toContainText(/running|completed/i)
    await expect(statusChip).toContainText(/completed/i)

    await expect(page.getByText(/^Job ID\b/)).toBeVisible()

    const tsvLink = page.getByRole('link', { name: /^Open TSV Output$/i })
    const reportLink = page.getByRole('link', { name: /^Open HTML Report$/i })

    await expect(tsvLink).toBeVisible()
    await expect(reportLink).toBeVisible()

    const tsvHref = await tsvLink.getAttribute('href')
    const reportHref = await reportLink.getAttribute('href')

    expect(tsvHref).toBeTruthy()
    expect(reportHref).toBeTruthy()

    const tsvResp = await request.get(tsvHref as string)
    const reportResp = await request.get(reportHref as string)

    expect(tsvResp.ok()).toBeTruthy()
    expect(reportResp.ok()).toBeTruthy()

    const reportText = await reportResp.text()
    expect(reportText).toContain('PlasFlow v2 Classification Report')
  })
})
