#!/usr/bin/env node
/**
 * 使用html-to-image库将HTML文件转换为PNG
 * 需要先安装: npm install html-to-image puppeteer
 */

const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

async function convertHtmlToPng(htmlFilePath, outputPath) {
    let browser;
    try {
        console.log(`🔄 转换: ${path.basename(htmlFilePath)}`);
        
        // 启动浏览器
        browser = await puppeteer.launch({
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        
        const page = await browser.newPage();
        
        // 设置视口大小
        await page.setViewport({ width: 1200, height: 800 });
        
        // 读取HTML文件内容
        const htmlContent = fs.readFileSync(htmlFilePath, 'utf8');
        
        // 修改HTML内容，添加html-to-image脚本
        const modifiedHtml = htmlContent.replace(
            '</head>',
            `
            <script src="https://cdn.jsdelivr.net/npm/html-to-image@1.11.11/dist/html-to-image.js"></script>
            </head>`
        );
        
        // 设置页面内容
        await page.setContent(modifiedHtml, { waitUntil: 'networkidle0' });
        
        // 等待页面完全加载
        await page.waitForTimeout(3000);
        
        // 执行截图脚本
        const screenshot = await page.evaluate(async () => {
            // 等待html-to-image库加载
            if (typeof htmlToImage === 'undefined') {
                throw new Error('html-to-image library not loaded');
            }
            
            // 找到主要内容区域
            const targetElement = document.body;
            
            try {
                // 使用html-to-image转换为PNG
                const dataUrl = await htmlToImage.toPng(targetElement, {
                    quality: 1.0,
                    pixelRatio: 2,
                    backgroundColor: 'white'
                });
                
                return dataUrl;
            } catch (error) {
                console.error('html-to-image conversion failed:', error);
                return null;
            }
        });
        
        if (screenshot) {
            // 将base64数据保存为PNG文件
            const base64Data = screenshot.replace(/^data:image\/png;base64,/, '');
            fs.writeFileSync(outputPath, base64Data, 'base64');
            console.log(`✅ 成功保存: ${path.basename(outputPath)}`);
            return true;
        } else {
            // 如果html-to-image失败，使用puppeteer截图作为备选
            console.log(`⚠️ html-to-image失败，使用puppeteer截图`);
            await page.screenshot({ 
                path: outputPath, 
                fullPage: true,
                type: 'png'
            });
            console.log(`✅ 备选方案成功: ${path.basename(outputPath)}`);
            return true;
        }
        
    } catch (error) {
        console.error(`❌ 转换失败 ${path.basename(htmlFilePath)}:`, error.message);
        return false;
    } finally {
        if (browser) {
            await browser.close();
        }
    }
}

async function findHtmlFiles() {
    const htmlFiles = [];
    const searchPaths = [
        'results/',
        'experiments/dataset_benchmarks/figure_reproduction/figure3_delayed_xor/outputs/figures/',
        'experiments/legacy_spikingjelly/original_experiments/figure_reproduction/figure3_delayed_xor/outputs/figures/',
        'experiments/dataset_benchmarks/temporal_dynamics/multi_timescale_xor/results/',
        'experiments/legacy_spikingjelly/original_experiments/temporal_dynamics/multi_timescale_xor/results/'
    ];
    
    for (const searchPath of searchPaths) {
        if (fs.existsSync(searchPath)) {
            const files = fs.readdirSync(searchPath);
            for (const file of files) {
                if (file.endsWith('.html')) {
                    htmlFiles.push(path.join(searchPath, file));
                }
            }
        }
    }
    
    return htmlFiles;
}

async function main() {
    console.log('🎨 HTML转PNG转换器 (使用html-to-image)');
    console.log('='.repeat(50));
    
    // 检查依赖
    try {
        require('puppeteer');
        console.log('✅ puppeteer 已安装');
    } catch (error) {
        console.log('❌ 需要安装 puppeteer: npm install puppeteer');
        process.exit(1);
    }
    
    // 查找HTML文件
    const htmlFiles = await findHtmlFiles();
    console.log(`📁 找到 ${htmlFiles.length} 个HTML文件`);
    
    if (htmlFiles.length === 0) {
        console.log('⚠️ 未找到HTML文件');
        return;
    }
    
    // 创建输出目录
    const outputDir = 'DH-SNN_Reproduction_Report/figures';
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // 选择重要的文件进行转换
    const importantFiles = htmlFiles.filter(file => {
        const filename = path.basename(file);
        return ['figure3_final', 'complete_figure4', 'performance_comparison', 'summary_dashboard']
            .some(keyword => filename.includes(keyword));
    });
    
    const filesToConvert = importantFiles.length > 0 ? importantFiles : htmlFiles.slice(0, 5);
    
    console.log(`🔄 转换 ${filesToConvert.length} 个重要文件...`);
    
    let successCount = 0;
    
    for (const htmlFile of filesToConvert) {
        const filename = path.basename(htmlFile, '.html');
        const outputPath = path.join(outputDir, `${filename}.png`);
        
        const success = await convertHtmlToPng(htmlFile, outputPath);
        if (success) {
            successCount++;
        }
        
        // 添加延迟避免资源冲突
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    console.log(`\n🎉 转换完成! 成功: ${successCount}/${filesToConvert.length}`);
    console.log(`📁 输出目录: ${outputDir}`);
    
    if (successCount > 0) {
        console.log('\n📋 生成的图片:');
        const pngFiles = fs.readdirSync(outputDir).filter(f => f.endsWith('.png'));
        pngFiles.forEach(file => {
            console.log(`  • ${file}`);
        });
    }
}

// 运行主函数
if (require.main === module) {
    main().catch(console.error);
}
