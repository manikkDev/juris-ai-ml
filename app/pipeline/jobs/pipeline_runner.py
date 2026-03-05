"""
Pipeline Runner
Orchestrates the complete data ingestion and processing pipeline
"""
import sys
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.pipeline.download.aws_downloader import IndianHighCourtDownloader, download_sample_data
from app.pipeline.extract.pdf_text_extractor import PDFTextExtractor
from app.pipeline.extract.ocr_fallback import extract_with_ocr_fallback
from app.pipeline.dataset.dataset_builder import DatasetBuilder
from app.pipeline.storage.dataset_store import DatasetStore
from app.utils.logger import logger


class PipelineRunner:
    """Orchestrate the complete data pipeline"""
    
    def __init__(
        self,
        pdf_dir: str = "data/raw/pdfs",
        text_dir: str = "data/intermediate/text",
        output_dir: str = "data/processed"
    ):
        """
        Initialize pipeline runner
        
        Args:
            pdf_dir: Directory for PDF files
            text_dir: Directory for extracted text
            output_dir: Directory for processed datasets
        """
        self.pdf_dir = Path(pdf_dir)
        self.text_dir = Path(text_dir)
        self.output_dir = Path(output_dir)
        
        # Initialize components
        self.downloader = IndianHighCourtDownloader(str(self.pdf_dir))
        self.text_extractor = PDFTextExtractor(str(self.text_dir))
        self.dataset_builder = DatasetBuilder(str(self.output_dir))
        self.dataset_store = DatasetStore(str(self.output_dir))
        
        logger.info("Initialized pipeline runner")
    
    def run_download(
        self,
        year: Optional[int] = None,
        court: Optional[str] = None,
        max_files: int = 100,
        use_sample: bool = False
    ) -> List[Path]:
        """
        Run download stage
        
        Args:
            year: Filter by year
            court: Filter by court
            max_files: Maximum files to download
            use_sample: Use sample data instead of AWS
        
        Returns:
            List of downloaded PDF paths
        """
        logger.info("=" * 60)
        logger.info("STAGE 1: DOWNLOAD")
        logger.info("=" * 60)
        
        if use_sample:
            logger.info("Using sample data for demonstration")
            pdf_files = download_sample_data(str(self.pdf_dir), num_files=min(max_files, 10))
        else:
            pdf_files = self.downloader.download_batch(
                year=year,
                court=court,
                max_files=max_files
            )
        
        logger.info(f"Downloaded {len(pdf_files)} PDF files")
        return pdf_files
    
    def run_extraction(
        self,
        pdf_files: Optional[List[Path]] = None,
        use_ocr: bool = False
    ) -> List[Path]:
        """
        Run text extraction stage
        
        Args:
            pdf_files: List of PDF files (None to process all in pdf_dir)
            use_ocr: Use OCR for extraction
        
        Returns:
            List of extracted text file paths
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: TEXT EXTRACTION")
        logger.info("=" * 60)
        
        # Get PDF files if not provided
        if pdf_files is None:
            pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        logger.info(f"Extracting text from {len(pdf_files)} PDFs")
        
        text_files = []
        
        with tqdm(total=len(pdf_files), desc="Extracting text") as pbar:
            for pdf_file in pdf_files:
                try:
                    if use_ocr:
                        # Use OCR extraction
                        text = extract_with_ocr_fallback(pdf_file, use_ocr=True)
                        
                        if text:
                            text_filename = pdf_file.stem + ".txt"
                            text_path = self.text_extractor.save_extracted_text(text, text_filename)
                            text_files.append(text_path)
                    else:
                        # Use regular extraction
                        result = self.text_extractor.extract_and_save(pdf_file)
                        
                        if result:
                            text_files.append(Path(result['text_path']))
                
                except Exception as e:
                    logger.error(f"Error extracting {pdf_file.name}: {str(e)}")
                
                pbar.update(1)
        
        logger.info(f"Successfully extracted {len(text_files)} text files")
        return text_files
    
    def run_dataset_building(
        self,
        text_files: Optional[List[Path]] = None,
        output_filename: str = "dataset.csv"
    ) -> Optional[Dict]:
        """
        Run dataset building stage
        
        Args:
            text_files: List of text files (None to process all in text_dir)
            output_filename: Output dataset filename
        
        Returns:
            Dataset statistics or None
        """
        logger.info("=" * 60)
        logger.info("STAGE 3: DATASET BUILDING")
        logger.info("=" * 60)
        
        # Get text files if not provided
        if text_files is None:
            text_files = list(self.text_dir.glob("*.txt"))
        
        logger.info(f"Building dataset from {len(text_files)} text files")
        
        # Build dataset
        df = self.dataset_builder.build_dataset_from_texts(
            text_files,
            output_filename=output_filename
        )
        
        if df.empty:
            logger.error("Failed to build dataset")
            return None
        
        # Validate dataset
        validation = self.dataset_builder.validate_dataset(df)
        
        if not validation['is_valid']:
            logger.warning(f"Dataset validation warnings: {validation['warnings']}")
        
        # Save with versioning
        self.dataset_store.save_dataset(
            df,
            filename=output_filename,
            description=f"Pipeline run - {len(df)} samples"
        )
        
        # Get statistics
        stats = self.dataset_builder.get_dataset_statistics(df)
        
        logger.info("Dataset statistics:")
        logger.info(f"  Total cases: {stats['total_cases']}")
        logger.info(f"  Avg case age: {stats['avg_case_age']:.1f} days")
        logger.info(f"  Avg adjournments: {stats['avg_adjournments']:.2f}")
        logger.info(f"  Avg hearings: {stats['avg_hearings']:.2f}")
        
        return stats
    
    def run_complete_pipeline(
        self,
        year: Optional[int] = None,
        court: Optional[str] = None,
        max_files: int = 100,
        use_sample: bool = False,
        use_ocr: bool = False,
        output_filename: str = "dataset.csv"
    ) -> Dict:
        """
        Run complete pipeline from download to dataset
        
        Args:
            year: Filter by year
            court: Filter by court
            max_files: Maximum files to download
            use_sample: Use sample data
            use_ocr: Use OCR for extraction
            output_filename: Output dataset filename
        
        Returns:
            Pipeline results dictionary
        """
        logger.info("=" * 60)
        logger.info("JURIS AI - DATA PIPELINE")
        logger.info("=" * 60)
        
        results = {
            'success': False,
            'stages': {},
            'errors': []
        }
        
        try:
            # Stage 1: Download
            pdf_files = self.run_download(
                year=year,
                court=court,
                max_files=max_files,
                use_sample=use_sample
            )
            
            results['stages']['download'] = {
                'files_downloaded': len(pdf_files),
                'success': len(pdf_files) > 0
            }
            
            if not pdf_files:
                results['errors'].append("No PDF files downloaded")
                return results
            
            # Stage 2: Text Extraction
            text_files = self.run_extraction(pdf_files, use_ocr=use_ocr)
            
            results['stages']['extraction'] = {
                'files_extracted': len(text_files),
                'success': len(text_files) > 0
            }
            
            if not text_files:
                results['errors'].append("No text files extracted")
                return results
            
            # Stage 3: Dataset Building
            stats = self.run_dataset_building(text_files, output_filename)
            
            results['stages']['dataset_building'] = {
                'success': stats is not None,
                'statistics': stats
            }
            
            if stats is None:
                results['errors'].append("Dataset building failed")
                return results
            
            # Pipeline completed successfully
            results['success'] = True
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            results['errors'].append(str(e))
        
        return results
    
    def get_pipeline_status(self) -> Dict:
        """
        Get current pipeline status
        
        Returns:
            Status dictionary
        """
        status = {
            'pdf_files': len(list(self.pdf_dir.glob("*.pdf"))),
            'text_files': len(list(self.text_dir.glob("*.txt"))),
            'dataset_versions': len(self.dataset_store.list_versions()),
            'current_version': self.dataset_store.get_current_version(),
        }
        
        return status


def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description="Juris AI Data Pipeline")
    
    parser.add_argument('--year', type=int, help='Filter by year')
    parser.add_argument('--court', type=str, help='Filter by court name')
    parser.add_argument('--max-files', type=int, default=10, help='Maximum files to process')
    parser.add_argument('--use-sample', action='store_true', help='Use sample data instead of real AWS data')
    parser.add_argument('--use-ocr', action='store_true', help='Use OCR for extraction')
    parser.add_argument('--output', type=str, default='dataset.csv', help='Output filename')
    
    args = parser.parse_args()
    
    # Run pipeline
    runner = PipelineRunner()
    
    results = runner.run_complete_pipeline(
        year=args.year,
        court=args.court,
        max_files=args.max_files,
        use_sample=args.use_sample,
        use_ocr=args.use_ocr,
        output_filename=args.output
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(f"Success: {results['success']}")
    
    if results['errors']:
        print(f"\nErrors: {', '.join(results['errors'])}")
    
    for stage, data in results.get('stages', {}).items():
        print(f"\n{stage.upper()}:")
        for key, value in data.items():
            print(f"  {key}: {value}")
    
    print("=" * 60)
    
    return 0 if results['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
